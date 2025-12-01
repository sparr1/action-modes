class SupportClassifier():
    def __init__(self, rule):
        self.rule = rule
    
    def __call__(self, obs):
        return self.rule(obs)


class UniversalSupport(SupportClassifier):
    def __init__(self):
        super().__init__(lambda x: True)
 