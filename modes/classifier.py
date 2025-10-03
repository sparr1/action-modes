class SupportClassifier():
    def __init__(self, rule):
        self.rule = rule

    def is_supported(self, obs):
        return self.rule(obs)


class UniversalSupport(SupportClassifier):
    def __init__(self):
        super().__init__(lambda x: True)
 