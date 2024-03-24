

import os,sys

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException


try:
    print(d = 1/0)

except Exception as e:
    logging.error(e)
    raise CustomException(e,sys)

