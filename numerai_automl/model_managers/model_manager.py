from ..model_managers.abstract_model_manager import AbstractModelManager
from ..config import TARGET_CANDIDATES


class ModelManager(AbstractModelManager):
    

    def __init__(self, targets: List[str]):
        self.targets = targets
        self.models = {}
        self.get_models()  # Initialize models upon creation

    def get_models(self):

        for 
        """
        Initialize and return a dictionary of scikit-learn models.
        """
        if not self.models:
            self.models = {
                'logistic_regression': LogisticRegression(),
                'decision_tree': DecisionTreeClassifier(),
                'svc': SVC()
            }
        return self.models