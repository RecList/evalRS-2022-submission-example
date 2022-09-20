"""

    Entry point to run the submission code: this is a modified version from the template
    provided in: https://github.com/RecList/evalRS-CIKM-2022/blob/main/submission.py

    The main change is the model class init, since the custom model need some parameters.

    Check the original template and evalRS repo (https://github.com/RecList/evalRS-CIKM-2022) for
    the fully commented version, and detailed instructions on the competition.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv('upload.env', verbose=True)


EMAIL = os.getenv('EMAIL')
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')


# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    # import my custom recList, containing the custom test
    from evaluation.EvalRSRecList import myRecList
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.MyModel import MyTwoTowerModel
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    dataset = ChallengeDataset(force_download=True)
    print('\n\n==== Init runner at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    my_model = MyTwoTowerModel(
        items_df=dataset.df_tracks,
        users_df=dataset.df_users,
        
        # Training hparams
        epochs=1,
        train_batch_size=8192,
        lr=1e-3,
        lr_decay_steps=100,
        lr_decay_rate=0.96,
        label_smoothing=0.0,
        
        # Model hparams
        logq_correction_factor=1.0,
        embeddings_l2_reg=1e-5,
        logits_temperature=1.8,
        tt_mlp_layers=[128,64],
        tt_mlp_activation="relu",
        tt_mlp_dropout=0.3,
        tt_mlp_l2_reg=5e-5,
        tt_infer_embedding_sizes_multiplier=2.0
    )
    # run evaluation with your model
    runner.evaluate(
        model=my_model,
        custom_RecList=myRecList # pass my custom reclist to the runner!
        )
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
