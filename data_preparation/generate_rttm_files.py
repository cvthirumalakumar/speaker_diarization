import re
import textgrid
import os
from glob import glob

def get_utterances_textgrid(tg_path):
    tg = textgrid.TextGrid()
    tg.read(tg_path)
    utterances = []
    for tier in tg.tiers:
        for interval in tier.intervals:
            if len(interval.mark) > 0:
                utterances.append({'text': interval.mark,
                                   'from': interval.minTime,
                                   'to': interval.maxTime})
    return utterances

def strip_transcript_tags(text):
    tags = ["<UNSURE>", "</UNSURE>", "<UNIN/>", "<INAUDIBLE_SPEECH/>"]
    for t in tags:
        text = text.replace(t, "")
    text = re.sub(r'\s+', ' ', text)
    text = text.lstrip().rstrip()
    return text

def get_combined_transcript(transcript_path_doctor, transcript_path_patient):
    utterances_doctor = get_utterances_textgrid(transcript_path_doctor)
    utterances_patient = get_utterances_textgrid(transcript_path_patient)
    for u in utterances_doctor:
        u['speaker'] = 'Doctor'
    for u in utterances_patient:
        u['speaker'] = 'Patient'
    combined_utterances = utterances_doctor + utterances_patient
    combined_utterances.sort(key=lambda x: x['from'])
    return combined_utterances

def __parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate rttm files for speaker diarisation')
    parser.add_argument('--transcript_path',
                        help='Folder containing TextGrid transcripts')
    parser.add_argument('--output_path', help='Output folder')
    return parser.parse_args()

def main():
    args = __parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    transcript_paths_doctor = glob(f'{args.transcript_path}/*doctor.TextGrid')
    for path_doctor in transcript_paths_doctor:
        print(path_doctor)
        path_patient = path_doctor.replace('doctor', 'patient')
        text_grid_file_name = path_patient.split("/")[-1].split(".")[0]
        file_id = "_".join(text_grid_file_name.split("_")[:-1])
        file_name = file_id+".rttm"
        output_path_rttm = os.path.join(args.output_path,
                                              file_name)
        combined_transcript = get_combined_transcript(path_doctor,
                                                       path_patient)
        print(output_path_rttm)
        with open(output_path_rttm, 'w') as f:
            for chunk in combined_transcript:
                if strip_transcript_tags(chunk['text']) != "":
                    f.write(f"SPEAKER {file_id} 1 {chunk['from']} {chunk['to']-chunk['from']} <NA> <NA> {chunk['speaker']} <NA> <NA>\n")


if __name__ == '__main__':
    main()
    
## Usage
## python generate_rttm_files.py --transcript_path=<text grids path> --output_path=<output rttm path>

