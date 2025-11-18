.PHONY: clean_positives clean_negatives clean preprocess run_simple
clean:
	rm -r data/tf_sites

clean_positives:
	rm -r data/tf_sites/*/positive_examples.txt

clean_negatives:
	rm -r data/tf_sites/*/negative_examples.txt

preprocess:
	python src/preprocess.py

run_simple:
	python src/main.py --config real_configs/simple.yaml

preprocess_ctcf:
	python src/preprocess.py --tf CTCF --pwm_file ./data/factorbookMotifPwm.txt
