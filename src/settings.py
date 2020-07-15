import sys
class AugmentationType:
    def __init__(self, aug_type, sigma=0.1):
        self.type = aug_type
        self.sigma = 0
        self.noise=False
        if aug_type == 0:
            self.type_text = "Augmentation Type: No Augmentation"
        elif aug_type == 1:
            self.type_text = "Augmentation Type: Duplication"
        elif aug_type == 2:
            self.type_text = "Augmentation Type: Duplication with Noise (Sigma = " + str(sigma) + ")"
            self.noise=True
            self.sigma=sigma
        elif aug_type == 3:
            self.type_text = "Augmentation Type: SMOTE"
        else:
            print("Incorrect augmentation type given as input, only 0-3 is accepted")
            sys.exit()