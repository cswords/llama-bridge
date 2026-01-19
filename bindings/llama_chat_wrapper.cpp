/**
 * Llama Chat Bindings
 *
 * pybind11 wrapper for llama.cpp's chat functionality.
 * Exposes chat template handling and tool call parsing to Python.
 *
 * Note: This is a minimal binding focused on chat templates.
 * Full inference support requires additional work.
 */

#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

namespace py = pybind11;

/**
 * Minimal wrapper for testing chat template functionality.
 * Full implementation requires more careful API integration.
 */
class LlamaChatWrapper {
public:
  LlamaChatWrapper(const std::string &model_path) : model_path_(model_path) {
    // Initialize llama backend
    llama_backend_init();

    // Load model with default params
    llama_model_params model_params = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!model_) {
      llama_backend_free();
      throw std::runtime_error("Failed to load model: " + model_path);
    }

    // Initialize chat templates from model
    templates_ = common_chat_templates_init(model_, "", "", "");

    if (!templates_) {
      llama_model_free(model_);
      llama_backend_free();
      throw std::runtime_error("Failed to initialize chat templates");
    }

    loaded_ = true;
  }

  ~LlamaChatWrapper() {
    if (model_) {
      llama_model_free(model_);
    }
    llama_backend_free();
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

    // Convert Python messages to common_chat_msg
    for (const auto &msg : messages) {
      common_chat_msg chat_msg;

      if (msg.count("role")) {
        chat_msg.role = msg.at("role").cast<std::string>();
      }
      if (msg.count("content")) {
        auto content = msg.at("content");
        if (py::isinstance<py::str>(content)) {
          chat_msg.content = content.cast<std::string>();
        }
      }
      if (msg.count("tool_call_id")) {
        chat_msg.tool_call_id = msg.at("tool_call_id").cast<std::string>();
      }
      if (msg.count("tool_name")) {
        chat_msg.tool_name = msg.at("tool_name").cast<std::string>();
      }

      // Handle tool_calls if present
      if (msg.count("tool_calls")) {
        auto tc_obj = msg.at("tool_calls");
        if (py::isinstance<py::list>(tc_obj)) {
          for (auto tc : tc_obj.cast<py::list>()) {
            auto tc_dict = tc.cast<py::dict>();
            common_chat_tool_call tool_call;
            if (tc_dict.contains("name")) {
              tool_call.name = tc_dict["name"].cast<std::string>();
            }
            if (tc_dict.contains("arguments")) {
              tool_call.arguments = tc_dict["arguments"].cast<std::string>();
            }
            if (tc_dict.contains("id")) {
              tool_call.id = tc_dict["id"].cast<std::string>();
            }
            chat_msg.tool_calls.push_back(tool_call);
          }
        }
      }

      inputs.messages.push_back(chat_msg);
    }

    // Convert Python tools to common_chat_tool
    for (const auto &tool : tools) {
      common_chat_tool chat_tool;
      if (tool.count("name")) {
        chat_tool.name = tool.at("name");
      }
      if (tool.count("description")) {
        chat_tool.description = tool.at("description");
      }
      if (tool.count("parameters")) {
        chat_tool.parameters = tool.at("parameters");
      }
      inputs.tools.push_back(chat_tool);
    }

    // Apply template
    common_chat_params params =
        common_chat_templates_apply(templates_.get(), inputs);

    // Return result
    py::dict result;
    result["prompt"] = params.prompt;
    result["grammar"] = params.grammar;
    result["format"] = common_chat_format_name(params.format);

    py::list stops;
    for (const auto &stop : params.additional_stops) {
      stops.append(stop);
    }
    result["additional_stops"] = stops;

    return result;
  }

  /**
   * Parse model response to extract text and tool calls.
   */
  py::dict parse_response(const std::string &response_text,
                          bool is_partial = false) {
    // Get format from templates
    common_chat_templates_inputs dummy_inputs;
    common_chat_params params =
        common_chat_templates_apply(templates_.get(), dummy_inputs);

    common_chat_syntax syntax;
    syntax.format = params.format;
    syntax.parse_tool_calls = true;

    // Parse the response
    common_chat_msg parsed =
        common_chat_parse(response_text, is_partial, syntax);

    // Convert to Python dict
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
   * Check if model is loaded.
   */
  bool is_loaded() const { return loaded_; }

  /**
   * Get model info.
   */
  py::dict get_model_info() const {
    py::dict info;
    info["path"] = model_path_;
    info["loaded"] = loaded_;
    if (model_) {
      const llama_vocab *vocab = llama_model_get_vocab(model_);
      info["n_vocab"] = llama_vocab_n_tokens(vocab);
      info["n_ctx_train"] = llama_model_n_ctx_train(model_);
    }
    return info;
  }

  /**
   * Get the chat template source.
   */
  std::string get_template_source() const {
    const char *src = common_chat_templates_source(templates_.get());
    return src ? std::string(src) : "";
  }

private:
  std::string model_path_;
  bool loaded_ = false;
  llama_model *model_ = nullptr;
  common_chat_templates_ptr templates_;
};

PYBIND11_MODULE(llama_chat, m) {
  m.doc() = "Llama.cpp chat bindings for Python";

  py::class_<LlamaChatWrapper>(m, "LlamaChatWrapper")
      .def(py::init<const std::string &>(), py::arg("model_path"),
           "Initialize wrapper with a GGUF model file")
      .def("apply_template", &LlamaChatWrapper::apply_template,
           py::arg("messages"),
           py::arg("tools") = std::vector<std::map<std::string, std::string>>(),
           py::arg("add_generation_prompt") = true,
           "Apply chat template to format messages into a prompt")
      .def("parse_response", &LlamaChatWrapper::parse_response,
           py::arg("response_text"), py::arg("is_partial") = false,
           "Parse model response to extract text and tool calls")
      .def("is_loaded", &LlamaChatWrapper::is_loaded,
           "Check if model is loaded")
      .def("get_model_info", &LlamaChatWrapper::get_model_info,
           "Get model information")
      .def("get_template_source", &LlamaChatWrapper::get_template_source,
           "Get the chat template source");

  // Module-level convenience functions
  m.def("version", []() { return "0.1.0"; });
}
