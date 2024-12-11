from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
import glob
from pyannote.database.util import load_rttm
import warnings
warnings.filterwarnings("ignore")



metric = DiarizationErrorRate()

ground_truth_rtms = "/media/nayan/g/kumar/speaker_diarisation/primock57/output/rttm"
reference_files = sorted(glob.glob(ground_truth_rtms+"/*.rttm"))

for folder in ['pyannote_3_1_preds','pyannote_2_1_preds','rev_v1_preds','rev_v2_preds','nemo_clustering_preds/pred_rttms','nemo_neural_preds/pred_rttms']:
    prediction_files = sorted(glob.glob(folder+"/*.rttm"))

    # Ensure both lists are aligned
    if len(reference_files) != len(prediction_files):
        raise ValueError("Number of reference and prediction files must match.")

    # Compute DER for each file
    for ref_file, pred_file in zip(reference_files, prediction_files):
        _, reference = load_rttm(ref_file).popitem()
        _, prediction = load_rttm(pred_file).popitem()
        metric(reference, prediction)
            # print(f"{ref_file} {der}")

    # Get overall DER
    overall_der = abs(metric)
    print(f"{folder}: Overall DER across all files: {overall_der:.2%}")
