.PHONY: clean_best_neg_seqs clean_fwd_best_neg_seqs clean_rev_best_neg_seqs
clean:
	rm -r data/tf_sites

clean_positives:
	rm -r data/tf_sites/*/positive/intervals.txt

clean_pos_seqs:
	rm -r data/tf_sites/*/positive/sequences.txt

clean_neg_seqs:
	rm -r data/tf_sites/*/negative/sequences.txt

clean_negatives:
	rm -r data/tf_sites/*/negative/intervals.txt

clean_best_neg_seqs: clean_fwd_best_neg_seqs clean_rev_best_neg_seqs

clean_fwd_best_neg_seqs:
	rm -r data/tf_sites/*/negative/best_negative_sequences.txt

clean_rev_best_neg_seqs:
	rm -r data/tf_sites/*/negative/reverse_best_negative_sequences.txt

clean_ml_flow:
	rm -r mlruns; rm mlflow.db

preprocess:
	python src/preprocess/preprocess.py

run_simple:
	python src/main.py --config real_configs/simple.yaml

preprocess_ctcf:
	python src/preprocess/preprocess.py --tf CTCF --pwm_file ./data/factorbookMotifPwm.txt

mlflow:
	mlflow server --port 5000
