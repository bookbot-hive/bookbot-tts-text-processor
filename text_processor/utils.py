import pandas as pd


class TextUtils:
    UNICODE_NORM_FORM = "NFKC"
    CUSTOM_TAGS = {}

    @classmethod
    def load_custom_tags(cls, csv_path):
        """Load custom tags from a CSV file."""
        if csv_path:
            df = pd.read_csv(csv_path)
            tags_dict = dict(zip(df["Tag name"], df["Number"]))
            return tags_dict
        return {}

    @classmethod
    def initialize_custom_tags(cls, csv_path):
        """Initialize the CUSTOM_TAGS dictionary."""
        if csv_path:
            cls.CUSTOM_TAGS = cls.load_custom_tags(csv_path)
        else:
            default_tags = {
                "handOverHeart": -1,
                "handRaiseMid": -2,
                "handRaiseHigh": -3,
                "handsBehindBack1": -4,
                "handsBehindBack2": -5,
                "headLean": -6,
                "raiseRightHand": -7,
                "raiseLeftHand": -8,
                "poked": -9,
                "idle": -10,
                "wave": -11,
                "listen": -12,
            }
            cls.CUSTOM_TAGS = default_tags

    @classmethod
    def get_custom_tags(cls):
        """Get the current custom tags dictionary."""
        return cls.CUSTOM_TAGS
