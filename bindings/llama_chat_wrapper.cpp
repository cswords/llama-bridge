/**
 * Copyright (c) 2026 Llama-Bridge Authors.
 * This software is released under the GNU General Public License v3.0.
 * See the LICENSE file in the project root for full license information.
 *
 * Llama Chat Bindings
 *
 * pybind11 wrapper for llama.cpp's chat functionality.
 * Exposes chat template handling and inference to Python.
 *
 * Supports multiple contexts (caches) for KV cache isolation.
 */

#include <iostream>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

namespace py = pybind11;

/**
 * State for a single context (cache).
 * Each context has its own KV cache and token history.
 */
struct ContextState {
  llama_context *ctx = nullptr;
  llama_sampler *sampler = nullptr;
  std::vector<llama_token> previous_tokens;
  int n_past = 0;
  int n_prompt_tokens = 0;
  int n_gen_tokens = 0;
};

/**
 * Wrapper class for llama.cpp chat functionality.
 * Supports multiple contexts for KV cache isolation.
 */
class LlamaChatWrapper {
public:
  LlamaChatWrapper(const std::string &model_path, int32_t n_ctx,
                   int32_t n_batch, int32_t n_ubatch, int32_t n_threads,
                   bool flash_attn)
      : model_path_(model_path), n_ctx_default_(n_ctx), n_batch_(n_batch),
        n_ubatch_(n_ubatch), n_threads_(n_threads), flash_attn_(flash_attn) {
    // Initialize llama backend
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!model_) {
      llama_backend_free();
      throw std::runtime_error("Failed to load model: " + model_path);
    }

    vocab_ = llama_model_get_vocab(model_);

    // Initialize chat templates from model
    templates_ = common_chat_templates_init(model_, "", "", "");

    if (!templates_) {
      llama_model_free(model_);
      llama_backend_free();
      throw std::runtime_error("Failed to initialize chat templates");
    }

    // Resolve context size (default for new contexts)
    if (n_ctx_default_ <= 0) {
      n_ctx_default_ = llama_model_n_ctx_train(model_);
      // Cap at 128k by default
      if (n_ctx_default_ > 131072)
        n_ctx_default_ = 131072;
    }

    // Resolve batch size
    if (n_batch_ <= 0) {
      n_batch_ = std::min((uint32_t)n_ctx_default_, 512u);
    }
    if (n_batch_ > n_ctx_default_) {
      n_batch_ = n_ctx_default_;
    }

    // Resolve ubatch
    if (n_ubatch_ <= 0) {
      n_ubatch_ = (n_batch_ > 0) ? n_batch_ : 512;
    }

    // Create default context
    create_context("default", n_ctx_default_);
    active_context_ = "default";

    loaded_ = true;
  }

  ~LlamaChatWrapper() {
    // Free all contexts
    for (auto &pair : contexts_) {
      if (pair.second.sampler) {
        llama_sampler_free(pair.second.sampler);
      }
      if (pair.second.ctx) {
        llama_free(pair.second.ctx);
      }
    }
    contexts_.clear();

    if (model_) {
      llama_model_free(model_);
    }
    llama_backend_free();
  }

  /**
   * Create a new context (cache) with specified size.
   */
  void create_context(const std::string &name, int32_t n_ctx) {
    if (contexts_.count(name)) {
      std::cerr << "DEBUG: Context '" << name
                << "' already exists, skipping creation" << std::endl;
      return;
    }

    if (n_ctx <= 0) {
      n_ctx = n_ctx_default_;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch_;
    ctx_params.n_ubatch = n_ubatch_;
    ctx_params.flash_attn_type = flash_attn_ ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                             : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.n_threads = (n_threads_ > 0) ? n_threads_ : 8;
    ctx_params.n_threads_batch = (n_threads_ > 0) ? n_threads_ : 8;
    ctx_params.n_seq_max = 1; // Single sequence per context

    llama_context *ctx = llama_init_from_model(model_, ctx_params);
    if (!ctx) {
      throw std::runtime_error("Failed to create context: " + name);
    }

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    ContextState state;
    state.ctx = ctx;
    state.sampler = sampler;
    contexts_[name] = state;

    std::cerr << "DEBUG: Created context '" << name << "' with n_ctx=" << n_ctx
              << std::endl;
  }

  /**
   * Select active context for inference.
   */
  void select_context(const std::string &name) {
    if (!contexts_.count(name)) {
      throw std::runtime_error("Context not found: " + name);
    }
    active_context_ = name;
    std::cerr << "DEBUG: Selected context '" << name << "'" << std::endl;
  }

  /**
   * Get list of available context names.
   */
  std::vector<std::string> list_contexts() const {
    std::vector<std::string> names;
    for (const auto &pair : contexts_) {
      names.push_back(pair.first);
    }
    return names;
  }

  /**
   * Apply chat template to format messages into a prompt.
   */
  py::dict
  apply_template(const std::vector<std::map<std::string, py::object>> &messages,
                 const std::vector<std::map<std::string, std::string>> &tools,
                 bool add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;

    for (const auto &msg : messages) {
      common_chat_msg chat_msg;
      if (msg.count("role"))
        chat_msg.role = msg.at("role").cast<std::string>();
      if (msg.count("content")) {
        auto content = msg.at("content");
        if (py::isinstance<py::str>(content))
          chat_msg.content = content.cast<std::string>();
      }
      if (msg.count("tool_call_id"))
        chat_msg.tool_call_id = msg.at("tool_call_id").cast<std::string>();
      if (msg.count("tool_name"))
        chat_msg.tool_name = msg.at("tool_name").cast<std::string>();

      if (msg.count("tool_calls")) {
        auto tc_obj = msg.at("tool_calls");
        if (py::isinstance<py::list>(tc_obj)) {
          for (auto tc : tc_obj.cast<py::list>()) {
            auto tc_dict = tc.cast<py::dict>();
            common_chat_tool_call tool_call;
            if (tc_dict.contains("name"))
              tool_call.name = tc_dict["name"].cast<std::string>();
            if (tc_dict.contains("arguments"))
              tool_call.arguments = tc_dict["arguments"].cast<std::string>();
            if (tc_dict.contains("id"))
              tool_call.id = tc_dict["id"].cast<std::string>();
            chat_msg.tool_calls.push_back(tool_call);
          }
        }
      }
      inputs.messages.push_back(chat_msg);
    }

    for (const auto &tool : tools) {
      common_chat_tool chat_tool;
      if (tool.count("name"))
        chat_tool.name = tool.at("name");
      if (tool.count("description"))
        chat_tool.description = tool.at("description");
      if (tool.count("parameters"))
        chat_tool.parameters = tool.at("parameters");
      inputs.tools.push_back(chat_tool);
    }

    common_chat_params params =
        common_chat_templates_apply(templates_.get(), inputs);

    py::dict result;
    result["prompt"] = params.prompt;
    result["grammar"] = params.grammar;
    result["format"] = common_chat_format_name(params.format);

    py::list stops;
    for (const auto &stop : params.additional_stops)
      stops.append(stop);
    result["additional_stops"] = stops;

    return result;
  }

  /**
   * Parse model response.
   */
  py::dict parse_response(const std::string &response_text,
                          bool is_partial = false) {
    common_chat_templates_inputs dummy_inputs;
    common_chat_params params =
        common_chat_templates_apply(templates_.get(), dummy_inputs);

    common_chat_parser_params syntax(params);
    if (!params.parser.empty()) {
      syntax.parser.load(params.parser);
    }
    syntax.parse_tool_calls = true;
    syntax.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    common_chat_msg parsed;
    try {
      parsed = common_chat_parse(response_text, is_partial, syntax);
    } catch (const std::exception &e) {
      std::cerr << "WARN: parse_response failed: " << e.what()
                << " - falling back to raw" << std::endl;
      parsed.content = response_text;
    }

    py::dict result;
    result["content"] = parsed.content;
    result["reasoning_content"] = parsed.reasoning_content;

    py::list tool_calls;
    for (const auto &tc : parsed.tool_calls) {
      py::dict tc_dict;
      tc_dict["name"] = tc.name;
      tc_dict["arguments"] = tc.arguments;
      tc_dict["id"] = tc.id;
      tool_calls.append(tc_dict);
    }
    result["tool_calls"] = tool_calls;

    return result;
  }

  /**
   * Initialize inference with a prompt on the active context.
   */
  void init_inference(const std::string &prompt, int max_tokens = 0) {
    ContextState &state = contexts_.at(active_context_);
    llama_context *ctx = state.ctx;

    // Tokenize
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);

    // Context Overflow Check
    int32_t n_ctx = llama_n_ctx(ctx);
    int32_t n_prompt = tokens.size();
    if (n_prompt + max_tokens > n_ctx) {
      std::string msg = "ContextLimitExceeded: Request (" +
                        std::to_string(n_prompt + max_tokens) +
                        " tokens) exceeds context limit (" +
                        std::to_string(n_ctx) + ")";
      throw std::runtime_error(msg);
    }

    // Find common prefix with previously cached tokens
    size_t n_keep = 0;
    while (n_keep < tokens.size() && n_keep < state.previous_tokens.size() &&
           tokens[n_keep] == state.previous_tokens[n_keep]) {
      n_keep++;
    }

    // Invalidate KV cache beyond n_keep
    llama_memory_t mem = llama_get_memory(ctx);
    bool rm_ok = llama_memory_seq_rm(mem, 0, (llama_pos)n_keep, -1);
    if (!rm_ok) {
      std::cerr << "WARNING: llama_memory_seq_rm(0, " << n_keep
                << ", -1) returned false!" << std::endl;
      llama_memory_seq_rm(mem, -1, (llama_pos)n_keep, -1);
    }

    // Handle full cache hit
    if (n_keep == tokens.size() && n_keep > 0) {
      n_keep--;
      llama_memory_seq_rm(mem, 0, (llama_pos)n_keep, -1);
      std::cerr << "DEBUG: [" << active_context_
                << "] Prompt cache full hit, re-decoding last token at pos "
                << n_keep << std::endl;
    } else {
      std::cerr << "DEBUG: [" << active_context_
                << "] Prompt cache partial hit: n_keep=" << n_keep
                << " tokens=" << tokens.size() << std::endl;
    }

    // Check context space
    int32_t n_ctx_total = llama_n_ctx(ctx);
    int32_t max_pos = (int32_t)llama_memory_seq_pos_max(mem, 0);
    int32_t n_used = (max_pos >= 0) ? (max_pos + 1) : 0;
    int32_t n_needed = (int32_t)tokens.size() - (int32_t)n_keep;

    if (n_used + n_needed > n_ctx_total) {
      std::cerr << "CRITICAL: [" << active_context_
                << "] Not enough KV cache space. Used=" << n_used
                << " Need=" << n_needed << " Total=" << n_ctx_total
                << std::endl;
      throw std::runtime_error("Context window exceeded (KV Cache Full)");
    }

    // Update usage
    state.n_prompt_tokens = tokens.size();
    state.n_gen_tokens = 0;

    // Decode in batches only the NEW parts
    int32_t n_batch_val = (int32_t)llama_n_batch(ctx);

    for (size_t i = n_keep; i < tokens.size(); i += n_batch_val) {
      int32_t n_eval = (int32_t)tokens.size() - i;
      if (n_eval > n_batch_val) {
        n_eval = n_batch_val;
      }

      llama_batch batch = llama_batch_init(n_eval, 0, 1);
      for (int32_t j = 0; j < n_eval; j++) {
        common_batch_add(batch, tokens[i + j], (int32_t)(i + j), {0}, false);
      }

      // Only request logits for the last token of the entire prompt
      if (i + n_eval == tokens.size()) {
        batch.logits[batch.n_tokens - 1] = true;
      }

      if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to decode prompt");
      }
      llama_batch_free(batch);
    }

    state.n_past = tokens.size();
    state.previous_tokens = tokens;
  }

  /**
   * Get next token from active context. Returns empty string if EOS.
   */
  std::string get_next_token() {
    ContextState &state = contexts_.at(active_context_);
    llama_context *ctx = state.ctx;
    llama_sampler *sampler = state.sampler;

    // Sample
    llama_token id = llama_sampler_sample(sampler, ctx, -1);

    if (llama_vocab_is_eog(vocab_, id)) {
      return "";
    }

    state.n_gen_tokens++;

    // Convert to piece
    std::string piece = common_token_to_piece(ctx, id, true);

    // Decode this token for next turn
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, id, state.n_past, {0}, true);

    if (llama_decode(ctx, batch) != 0) {
      llama_batch_free(batch);
      return ""; // Treat as error/EOS
    }
    llama_batch_free(batch);

    state.n_past++;
    state.previous_tokens.push_back(id);
    return piece;
  }

  /**
   * Simple non-streaming generate.
   */
  std::string generate(const std::string &prompt, int max_tokens = 4096) {
    init_inference(prompt, max_tokens);
    std::string result;
    for (int i = 0; i < max_tokens; i++) {
      std::string token = get_next_token();
      if (token.empty())
        break;
      result += token;
    }
    return result;
  }

  py::dict get_model_info() const {
    py::dict info;
    info["path"] = model_path_;
    info["loaded"] = loaded_;
    if (model_) {
      info["n_vocab"] = llama_vocab_n_tokens(vocab_);
      info["n_ctx_train"] = llama_model_n_ctx_train(model_);
    }
    // Include context info
    py::list ctx_list;
    for (const auto &pair : contexts_) {
      py::dict ctx_info;
      ctx_info["name"] = pair.first;
      if (pair.second.ctx) {
        ctx_info["n_ctx"] = llama_n_ctx(pair.second.ctx);
      }
      ctx_list.append(ctx_info);
    }
    info["contexts"] = ctx_list;
    info["active_context"] = active_context_;
    return info;
  }

  py::dict get_usage() const {
    const ContextState &state = contexts_.at(active_context_);
    py::dict usage;
    usage["context"] = active_context_;
    usage["prompt_tokens"] = state.n_prompt_tokens;
    usage["completion_tokens"] = state.n_gen_tokens;
    usage["total_tokens"] = state.n_prompt_tokens + state.n_gen_tokens;
    return usage;
  }

private:
  std::string model_path_;
  bool loaded_ = false;
  llama_model *model_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  common_chat_templates_ptr templates_;

  // Default parameters for new contexts
  int32_t n_ctx_default_;
  int32_t n_batch_;
  int32_t n_ubatch_;
  int32_t n_threads_;
  bool flash_attn_;

  // Multi-context support
  std::unordered_map<std::string, ContextState> contexts_;
  std::string active_context_;
};

PYBIND11_MODULE(llama_chat, m) {
  m.doc() = "Llama.cpp chat bindings for Python";
  py::class_<LlamaChatWrapper>(m, "LlamaChatWrapper")
      .def(py::init<const std::string &, int32_t, int32_t, int32_t, int32_t,
                    bool>(),
           py::arg("model_path"), py::arg("n_ctx") = 0, py::arg("n_batch") = 0,
           py::arg("n_ubatch") = 0, py::arg("n_threads") = 0,
           py::arg("flash_attn") = false)
      .def("create_context", &LlamaChatWrapper::create_context, py::arg("name"),
           py::arg("n_ctx") = 0)
      .def("select_context", &LlamaChatWrapper::select_context, py::arg("name"))
      .def("list_contexts", &LlamaChatWrapper::list_contexts)
      .def("apply_template", &LlamaChatWrapper::apply_template,
           py::arg("messages"), py::arg("tools"),
           py::arg("add_generation_prompt"))
      .def("parse_response", &LlamaChatWrapper::parse_response,
           py::arg("response_text"), py::arg("is_partial") = false)
      .def("init_inference", &LlamaChatWrapper::init_inference,
           py::arg("prompt"), py::arg("max_tokens") = 0)
      .def("get_next_token", &LlamaChatWrapper::get_next_token)
      .def("generate", &LlamaChatWrapper::generate, py::arg("prompt"),
           py::arg("max_tokens") = 4096)
      .def("get_model_info", &LlamaChatWrapper::get_model_info)
      .def("get_usage", &LlamaChatWrapper::get_usage);
  m.def("version", []() { return "0.2.0"; });
}
