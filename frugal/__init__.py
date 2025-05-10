def hello_world():
    print("✅ Module 'frugal' imported successfully")

from .models.ML_CountVect_NB.main import main as train_CountVect_MulitinomialNB
from .models.ML_TFIDF_NB.main import main as train_TFIDF_MulitinomialNB
from .models.RNN_basic.main import main as train_RNN_basic
from .models.RNN_LSTM.main import main as train_RNN_LSTM
