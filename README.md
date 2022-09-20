# EvalRS Submission Example
A sample repository with code, comments and videos to help with packaging the code for the final submission.

## Overview

This repository contains an easy to understand fake submission to the [EvalRS Data Challenge](https://github.com/RecList/evalRS-CIKM-2022). It serves three purposes:

* showing a clear project structure, based off the exact same code and abstractions in the official repository; remember that you're not allowed to modify the initial runner or recList (i.e. you should _extend_ it);

* showing how to instruct the organizers on how to best run the code to _reproduce the score on the leaderboard_: remember that if your score cannot be verified, you will be disqualified;

* giving you a step by step guide to verify that your submission is within the compute budget on the target machine (section at the bottom).

For the rules and guidelines on the compute budget and the target machine, please check the official [rules](https://github.com/RecList/evalRS-CIKM-2022).

This is a WIP. Come back often for updates!

## Project description (as if this was a submission!)

### Introduction

This is the submission of team ACME to the [EvalRS Data Challenge](https://github.com/RecList/evalRS-CIKM-2022).

### Instructions

_Preliminaries_


_Custom test_


### License (remember, needs to be an open soure license!)

All the project is covered under the attached MIT License.

## Making sure your code works on the target machine

### Manual Setup

If you have never done it before, we prepared this guide. The only pre-requisite is signing up for an AWS account (keep your credentials safe!).

_ATTENTION_: you will incur in costs by running a p3 machine - if you're just using for testing your compute budget, the expense is fairly low. You can check the on-demand price using AWS [official pricing](https://aws.amazon.com/it/ec2/pricing/on-demand/).

Once you have your account, follow these steps:

* Create an EC2 with the appropriate setup, and create a key to access it (store it safely): [video](https://watch.screencastify.com/v/mNmw8bR78bfFGjbGmBWA).

* When the instance is ready, you will see it in the [console](images/instance_ready.png).

* Find the instructions to connect to the instance through ssh and the key you created: [video](https://watch.screencastify.com/v/v6HLFmFgiqMe8PgBoU6h).

* Connect to the instance, clone your submission repo from Github (in this example, it is this very repo: `git clone https://github.com/RecList/evalRS-2022-submission-example`), and execute whatever preliminary steps you need. For example, make sure to install the dependencies and to create a `upload.env` file with your credentials (NOT SHOWN in the videos)).

* Please note that if you use a DL AMI the console will tell when the [DL-ready interpreter is](images/ec2.png). Make sure to use that to then launch your submission in the usual way, that is, in this case `/usr/local/bin/python3.9 submission.py`: [video](https://watch.screencastify.com/v/nDFJMcUcBb1dBTFrwcjH).

* When the eval loop has completed, you should see the usual console log LINK.

* When you're sure your compute budget tests have succeded, _shut down_ the machine to avoid incurring in additional costs!

#### FAQs

* _When I launch the `p3` machine, AWS says that I have a limit on the instance_. This may happen if this is your first AWS account: you need to follow the instructions on AWS console and open a ticket for them to unlock those machines for you, which are outside the free tier. Since it may take a while for them to do so, please try out this tutorial as soon as possible to make sure you have the setup ready for running your submission checks!

### Pulumi Setup

WIP