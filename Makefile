.PHONY: *

DROPBOX_DATASET := .dropbox_dataset

CLEARML_PROJECT_NAME := image_segmentation
CLEARML_DATASET_NAME := image_segmentation_dataset


migrate_dataset:
	# Migrate dataset to ClearML datasets.
	rm -R $(DROPBOX_DATASET) || true
	mkdir $(DROPBOX_DATASET)
	wget "https://www.dropbox.com/scl/fi/2ann71bbkh7wd8clzi0wc/image_segmentation.zip?rlkey=oqzmf6vprt29zvdmhj12iqnk6&dl=0" -O $(DROPBOX_DATASET)/dataset.zip
	unzip -q $(DROPBOX_DATASET)/dataset.zip -d $(DROPBOX_DATASET)
	rm $(DROPBOX_DATASET)/dataset.zip
	find $(DROPBOX_DATASET) -type f -name '.DS_Store' -delete
	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	clearml-data add --files $(DROPBOX_DATASET)
	clearml-data close --verbose
	rm -R $(DROPBOX_DATASET)

