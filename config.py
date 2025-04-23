import json


class ConfigError(Exception):
    pass


class Config:
    VALID_TASKS = {"joint", "multitask", "semantic"}
    VALID_MODELS = {"resnet", "efficientnet", "classic"}

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        self.denoised = config_data.get("denoised", False)
        self.cropped = config_data.get("cropped", False)
        self.clinical = config_data.get("clinical", False)
        self.model = config_data.get("model")
        self.task = config_data.get("task")
        self.batch_size = config_data.get("batch_size", 8)
        self.num_epochs = config_data.get("num_epochs", 100)
        self.learning_rate = config_data.get("learning_rate", 0.001)

        self._validate()
        self.dataset_path = self._assign_dataset_path()
        self.model = self._construct_model()

    def _validate(self):
        if self.task not in self.VALID_TASKS:
            raise ConfigError(f"Invalid task: {self.task}. Must be one of {self.VALID_TASKS}.")
        if self.model not in self.VALID_MODELS:
            raise ConfigError(f"Invalid model: {self.model}. Must be one of {self.VALID_MODELS}.")

    def _assign_dataset_path(self):
        if self.cropped and self.denoised:
            return "/path/to/cropped_denoised"
        elif self.cropped and not self.denoised:
            return "/path/to/cropped_not_denoised"
        elif not self.cropped and self.denoised:
            return "/path/to/not_cropped_denoised"
        else:
            return "/path/to/not_cropped_not_denoised"

    def _construct_model(self):
        # 1) what is the task?
        if self.task == "joint":
            return joint_architecture(self.model, self.clinical)
        elif self.task == "semantic":
            return semantic_architecture(self.model, self.clinical)
        elif self.task == "classification":
            return classification_architecture(self.model, self.clinical)
        else:
            raise ConfigError("Invalid task")

    def to_string(self):
        cropped_str = "cropped" if self.cropped else "notcropped"
        denoised_str = "denoised" if self.denoised else "notdenoised"
        clinical_str = "clinical" if self.clinical else "noclinical"
        return f"{cropped_str}_{denoised_str}_{clinical_str}_{self.model}_{self.task}"
