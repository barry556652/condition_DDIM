# from DiffusionFreeGuidence.TrainCondition import train, eval
# from DFG.TrainCondition import train, eval
from DDIM.TrainCondition import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "train", #train or eval
        "epoch": 100,
        "batch_size": 64,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attention_levels":[8, 4, 2],
        "num_labels": 8,
        "n_heads":4,
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 3e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "DDIM_steps":100,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_99_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
        eval(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
