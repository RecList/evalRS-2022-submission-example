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
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.MyModel import MyMFModel
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    dataset = ChallengeDataset()
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
    my_model = MyMFModel(
        items_df=dataset.df_tracks,
        users_df=dataset.df_users,
        # Training hparams
        epochs=5,
        train_batch_size=8192,
        lr=1e-3,
        lr_decay_steps=100,
        lr_decay_rate=0.96,
        label_smoothing=0.0,
        # Model hparams
        logq_correction_factor=1.0,
        embeddings_l2_reg=5e-6,
        logits_temperature=1.8,
        mf_dim=128,
    )
    # run evaluation with your model
    runner.evaluate(
        model=my_model
        )
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
