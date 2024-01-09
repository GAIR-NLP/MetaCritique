class OpenaiConfig:
    def __init__(self):
        self.model_type = "gpt-4"
        self.batch_size = 5

        # config your OpenAi key here
        self.org_id = None
        self.api_key = None
        self.api_base = None

        self.temperature = 0.0
        self.max_tokens = 512
        self.top_p = 1.0
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.request_timeout = 120

        # please note that only gpt-4-1106 has the following feature
        self.seed = None
