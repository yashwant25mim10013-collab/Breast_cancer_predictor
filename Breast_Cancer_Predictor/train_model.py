"""Train the models, pick the best one, and save it with joblib.  Run:  python train_model.py"""
import model_core as mc

if __name__ == "__main__":
    model, meta = mc.train_and_save(save=True)
    print("Cross-validation (training set only):")
    for name, s in meta["comparison"].items():
        print(f"  {name:20s} ROC-AUC {s['cv_roc_auc']:.4f} | accuracy {s['cv_accuracy']:.4f}")
    print(f"\nBest model: {meta['model_name']}")
    print("Test-set metrics:", {k: round(v, 4) for k, v in meta["test_metrics"].items()})
    print("Confusion matrix [rows = actual B, M]:", meta["confusion_matrix"])
    print("Saved: model/model.joblib and model/metadata.json")
