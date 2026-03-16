ENV_NAME := subcellvae

.PHONY: env env-cuda env-update notebook clean

env:
	conda env create -f environment.yml

env-cuda:
	conda env create -f environment-cuda.yml

env-update:
	conda env update -f environment.yml --prune

notebook:
	conda run -n $(ENV_NAME) jupyter lab --notebook-dir=notebook

clean:
	conda env remove -n $(ENV_NAME)



PYTHON = python

# ── Individual analysis runs ──────────────────────────────────────────────────

analysis_AE_lat8_UMAP_k6:
	$(PYTHON) scripts/run_analysis_pipeline.py \
		--config configs/analysis_config_AE_lat8_UMAP_k6.yaml

analysis_AE_lat16_UMAP_k6:
	$(PYTHON) scripts/run_analysis_pipeline.py \
		--config configs/analysis_config_AE_lat16_UMAP_k6.yaml

analysis_VAE_lat8_PHATE_k8:
	$(PYTHON) scripts/run_analysis_pipeline.py \
		--config configs/analysis_config_VAE_lat8_PHATE_k8.yaml

analysis_VAE_lat8_UMAP_PHATE_k6:
	$(PYTHON) scripts/run_analysis_pipeline.py \
		--config configs/analysis_config_VAE_lat8_UMAP_PHATE_k6.yaml

# ── Run all configs sequentially ──────────────────────────────────────────────

all_analysis:
	$(MAKE) analysis_AE_lat8_UMAP_k6
	$(MAKE) analysis_AE_lat16_UMAP_k6
	$(MAKE) analysis_VAE_lat8_PHATE_k8
	$(MAKE) analysis_VAE_lat8_UMAP_PHATE_k6

# ── Clean output folders ──────────────────────────────────────────────────────
# Update these paths to match your out_dir values in each config

RESULTS_DIR = /mnt/d/lding/CLS_GitHub/fa_patch_AE_clustering/results

clean_results:
	rm -rf $(RESULTS_DIR)/AE_lat8_UMAP_k6
	rm -rf $(RESULTS_DIR)/AE_lat16_UMAP_k6
	rm -rf $(RESULTS_DIR)/VAE_lat8_PHATE_k8
	rm -rf $(RESULTS_DIR)/VAE_lat8_UMAP_PHATE_k6

clean_all: clean_results

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  analysis_AE_lat8_UMAP_k6       Run AE lat8 UMAP k=6"
	@echo "  analysis_AE_lat16_UMAP_k6      Run AE lat16 UMAP k=6"
	@echo "  analysis_VAE_lat8_PHATE_k8     Run VAE lat8 PHATE k=8"
	@echo "  analysis_VAE_lat8_UMAP_PHATE_k6 Run VAE lat8 UMAP+PHATE k=6"
	@echo "  all_analysis                   Run all configs sequentially"
	@echo "  clean_results                  Delete all result folders"
	@echo "  help                           Show this message"

.PHONY: analysis_AE_lat8_UMAP_k6 analysis_AE_lat16_UMAP_k6 \
        analysis_VAE_lat8_PHATE_k8 analysis_VAE_lat8_UMAP_PHATE_k6 \
        all_analysis clean_results clean_all help
