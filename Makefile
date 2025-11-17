clean_positives:
	rm -r data/tf_sites/*/positive_examples.txt

clean_negatives:
	rm -r data/tf_sites/*/negative_examples.txt

clean:
	clean_positives
	clean_negatives

preprocess:
	python src/preprocess.py
