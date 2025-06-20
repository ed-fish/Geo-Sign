import jiwer
import re

def compute_wer(reference_file, hypothesis_file):
    # Read files while preserving empty lines
    with open(reference_file, 'r', encoding='utf-8') as f:
        refs = [line.strip() for line in f]
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        hyps = [line.strip() for line in f]

    # Validate line counts
    if len(refs) != len(hyps):
        raise ValueError(f"Mismatched lines: {len(refs)} references vs {len(hyps)} hypotheses")

    # Custom processing for Chinese text with space-separated words
    def process_line(line):
        # 1. Remove Chinese punctuation first
        line = re.sub(r'[。，、；：“”‘’！？（）《》【】]', '', line)
        # 2. Remove English punctuation (if any)
        line = re.sub(r'[^\w\s]', '', line)
        # 3. Remove extra spaces and normalize
        line = re.sub(r'\s+', ' ', line).strip()
        # 4. Split into words (return empty list if line is empty)
        return line.split() if line else []

    # Process both files without using jiwer's transformations
    processed_refs = [process_line(line) for line in refs]
    processed_hyps = [process_line(line) for line in hyps]

    # Calculate WER directly using the alignment
    wer = jiwer.wer(
        reference=processed_refs,
        hypothesis=processed_hyps,
        truth_transform=lambda x: x,  # Bypass jiwer's transforms
        hypothesis_transform=lambda x: x
    )
    
    print(f"Word Error Rate: {wer * 100:.2f}%")
    return wer
# Example usage
compute_wer("/home/ef0036/Projects/Uni-Sign-plus/Uni-Sign/out/test/dev_eval_references.txt", "/home/ef0036/Projects/Uni-Sign-plus/Uni-Sign/out/test/dev_eval_predictions.txt")