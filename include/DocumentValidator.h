#pragma once

#include <string>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

struct ModelResult {
    bool is_valid;
    float confidence;
    std::string model_name;
};

struct ValidationResult {
    bool final_verdict;
    ModelResult realism_model_1;
    ModelResult realism_model_2;
    ModelResult realism_model_3;
    bool realism_majority;
    ModelResult ocr_model;
    bool ocr_valid;
    std::string detailed_message;
};

class DocumentValidator {
public:
    DocumentValidator();
    ~DocumentValidator();

    bool loadModels(const std::string& project_root);
    ValidationResult validate(const std::string& document_path);
    ValidationResult getLastResult() const;
    std::string getValidationMessage() const;
    float getConfidence() const;

private:
    std::string project_root_;
    std::string python_executable_;
    bool models_loaded_;
    ValidationResult last_result_;
    
    std::string runPythonValidator(const std::string& document_path);
    ValidationResult parseJsonResult(const std::string& json_str);
    std::string findPythonExecutable();
};
