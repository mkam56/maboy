#include "../include/DocumentValidator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <cstdlib>
#include <array>
#include <memory>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

DocumentValidator::DocumentValidator() 
    : models_loaded_(false), project_root_(""), python_executable_("") {
    last_result_.final_verdict = false;
    last_result_.realism_majority = false;
    last_result_.ocr_valid = false;
    last_result_.detailed_message = "Models not loaded";
}

DocumentValidator::~DocumentValidator() {
}

bool DocumentValidator::loadModels(const std::string& project_root) {
    project_root_ = project_root;
    python_executable_ = findPythonExecutable();
    
    if (python_executable_.empty()) {
        last_result_.detailed_message = "Python not found";
        return false;
    }
    
    std::string validator_script = project_root + "/validator.py";
    std::ifstream script_file(validator_script);
    if (!script_file.good()) {
        last_result_.detailed_message = "validator.py not found";
        return false;
    }
    
    models_loaded_ = true;
    last_result_.detailed_message = "Ready for validation";
    return true;
}

ValidationResult DocumentValidator::validate(const std::string& document_path) {
    ValidationResult result;
    
    std::cout << "[C++] validate() called for: " << document_path << std::endl;
    
    if (!models_loaded_) {
        std::cout << "[C++] Models not loaded!" << std::endl;
        result.final_verdict = false;
        result.detailed_message = "Models not loaded";
        last_result_ = result;
        return result;
    }

    try {
        std::cout << "[C++] Calling runPythonValidator..." << std::endl;
        std::string json_output = runPythonValidator(document_path);
        std::cout << "[C++] Got output, parsing JSON..." << std::endl;
        result = parseJsonResult(json_output);
        std::cout << "[C++] Validation complete!" << std::endl;
        last_result_ = result;
        return result;
    } catch (const std::exception& e) {
        std::cout << "[C++] Exception: " << e.what() << std::endl;
        result.final_verdict = false;
        result.detailed_message = std::string("Validation error: ") + e.what();
        last_result_ = result;
        return result;
    }
}

ValidationResult DocumentValidator::getLastResult() const {
    return last_result_;
}

std::string DocumentValidator::getValidationMessage() const {
    return last_result_.detailed_message;
}

float DocumentValidator::getConfidence() const {
    float total = 0.0f;
    int count = 0;
    
    if (last_result_.realism_model_1.confidence > 0) {
        total += last_result_.realism_model_1.confidence;
        count++;
    }
    if (last_result_.realism_model_2.confidence > 0) {
        total += last_result_.realism_model_2.confidence;
        count++;
    }
    if (last_result_.realism_model_3.confidence > 0) {
        total += last_result_.realism_model_3.confidence;
        count++;
    }
    if (last_result_.ocr_model.confidence > 0) {
        total += last_result_.ocr_model.confidence;
        count++;
    }
    
    return count > 0 ? total / count : 0.0f;
}

std::string DocumentValidator::findPythonExecutable() {
    std::vector<std::string> candidates = {
        project_root_ + "/venv/bin/python3",
        project_root_ + "/venv/bin/python",
        project_root_ + "/.venv/bin/python3",
        project_root_ + "/.venv/bin/python",
        "python3",
        "python",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3"
    };
    
    for (const auto& candidate : candidates) {
        std::ifstream test(candidate);
        if (test.good()) {
            test.close();
            return candidate;
        }
        
        std::string cmd = "which " + candidate + " 2>/dev/null";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            char buffer[256];
            std::string result;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            int status = pclose(pipe);
            
            if (!result.empty() && result.back() == '\n') {
                result.pop_back();
            }
            
            if (status == 0 && !result.empty()) {
                return result;
            }
        }
    }
    
    return "python3";
}

std::string DocumentValidator::runPythonValidator(const std::string& document_path) {
    std::string validator_script = project_root_ + "/validator.py";
    std::ostringstream cmd;
    cmd << python_executable_ << " \"" << validator_script << "\" "
        << "\"" << document_path << "\" "
        << "--project-root \"" << project_root_ << "\"";
    
    std::cout << "[C++] Running command: " << cmd.str() << std::endl;
    
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::cout << "[C++] Failed to create pipe!" << std::endl;
        throw std::runtime_error("Failed to run Python validator");
    }
    
    std::cout << "[C++] Pipe created, reading output..." << std::endl;
    std::string result;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    
    int status = pclose(pipe);
    std::cout << "[C++] Python exited with status: " << status << std::endl;
    std::cout << "[C++] Output length: " << result.length() << " bytes" << std::endl;
    
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        result = result.substr(start, end - start + 1);
    }
    
    if (result.empty() || result.find("{") == std::string::npos) {
        throw std::runtime_error("Python validator returned no valid output");
    }
    
    return result;
}

ValidationResult DocumentValidator::parseJsonResult(const std::string& json_str) {
    ValidationResult result;
    
    try {
        auto j = json::parse(json_str);
        
        if (j.contains("error")) {
            result.final_verdict = false;
            result.detailed_message = j["error"].get<std::string>();
            return result;
        }
        
        result.final_verdict = j.value("final_verdict", false);
        result.realism_majority = j.value("realism_majority", false);
        result.ocr_valid = j.value("ocr_valid", false);
        result.detailed_message = j.value("detailed_message", "");
        
        if (j.contains("realism_results") && j["realism_results"].is_array()) {
            auto realism_array = j["realism_results"];
            
            if (realism_array.size() > 0) {
                auto r1 = realism_array[0];
                result.realism_model_1.is_valid = r1.value("is_valid", false);
                result.realism_model_1.confidence = r1.value("confidence", 0.0f);
                result.realism_model_1.model_name = r1.value("model_name", "Realism Model 1");
            }
            
            if (realism_array.size() > 1) {
                auto r2 = realism_array[1];
                result.realism_model_2.is_valid = r2.value("is_valid", false);
                result.realism_model_2.confidence = r2.value("confidence", 0.0f);
                result.realism_model_2.model_name = r2.value("model_name", "Realism Model 2");
            }
            
            if (realism_array.size() > 2) {
                auto r3 = realism_array[2];
                result.realism_model_3.is_valid = r3.value("is_valid", false);
                result.realism_model_3.confidence = r3.value("confidence", 0.0f);
                result.realism_model_3.model_name = r3.value("model_name", "Realism Model 3");
            }
        }
        
        if (j.contains("ocr_result")) {
            auto ocr = j["ocr_result"];
            result.ocr_model.is_valid = ocr.value("is_valid", false);
            result.ocr_model.confidence = ocr.value("confidence", 0.0f);
            result.ocr_model.model_name = ocr.value("model_name", "OCR Model");
        }
        
    } catch (const json::exception& e) {
        result.final_verdict = false;
        result.detailed_message = std::string("JSON parsing error: ") + e.what();
    }
    
    return result;
}
