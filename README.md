# JIONE: AN APPROACH FOR MERGING LARGE LANGUAGE MODELS VIA TEACHER–STUDENT PREDICTION REFINEMENT
This repository contains the replication package for the submitted paper, including code, datasets, and instructions for running the experiments.

1. Single Merging  
For people who do not have access to multiple GPUs to run experiments for several days, you can run JIONE only for the merging of the Phi-3-mini-4k-instruct (student) and Phi-3-mini-128k-instruct (teacher) models.  
This run takes around **14 hours** on a single NVIDIA T4 GPU.
To ensure exact reproducibility of this experiment, we used the specific `revision_id` of the models during our experiments.

## Installing the Required Packages
To set up the environment, create a conda environment and install the required packages using the `requirements.txt` file with te following commands:
```
cd jione_single_merging
conda create -n shared --file requirements.txt
conda activate shared
```

## Run JIONE code
To run the JIONE approach for a single merging, simply use the command `python run.py` and wait for the prediction process to finish. This will produce the merging results for each task.  
The generated answers are saved in the **outputs** folder. This folder is automatically created after the first prediction file is saved if it does not already exist.

## Evaluation and Results
Once the run ends, you should obtain the following evaluation results:
- **TruthfulQA:** 0.5177 Accuracy
- **MBPP:** 0.2844 ROUGE-Score
- **GSM8K:** 0.7672 Exact Match Accuracy (EMA)
- **SST-2:** 0.9670 Accuracy

The following outputs illustrate the use of JIONE on five samples from each dataset.
```
----------------------------- Starting predictions ---------------------
Prediction process for truthful_qa task: 100%|████| 5/5 [00:19<00:00,  3.97s/it]
100%|█████████████████████████████████████████| 5/5 [00:00<00:00, 166440.63it/s]
[OK] Saved 5 rows for task 'truthful_qa' to outputs/truthful_qa_phi128k(t)_phi4k(s)_prediction.csv 

[EVAL] 'truthful_qa' ==> 0.8 

Prediction process for mbpp task: 100%|███████████| 5/5 [01:09<00:00, 13.91s/it]
100%|███████████████████████████████████████████| 5/5 [00:00<00:00, 1033.54it/s]
[OK] Saved 5 rows for task 'mbpp' to outputs/mbpp_phi128k(t)_phi4k(s)_prediction.csv 

[EVAL] 'mbpp' ==> 0.16563364864109503 

Prediction process for gsm8k task: 100%|██████████| 5/5 [01:54<00:00, 22.97s/it]
100%|██████████████████████████████████████████| 5/5 [00:00<00:00, 16831.08it/s]
[OK] Saved 5 rows for task 'gsm8k' to outputs/gsm8k_phi128k(t)_phi4k(s)_prediction.csv 

[EVAL] 'gsm8k' ==> 0.8 

Prediction process for sst-2 task: 100%|██████████| 5/5 [00:26<00:00,  5.22s/it]
[OK] Saved 5 rows for task 'sst-2' to outputs/sst-2_phi128k(t)_phi4k(s)_prediction.csv 

[EVAL] 'sst-2' ==> 0.8 
```
## Kaggle Notebook
Kaggle provides free computing resources, including two NVIDIA T4 GPU instances, to help users run their experiments. Thus, if you do not have sufficient GPU resources to run on our work your own machine, you can reproduce and test it using this Kaggle notebook: [![Kaggle](https://kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/code/furelteguimene/llm-merging-latest)


<!-- ## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/tsotsa/jione.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/tsotsa/jione/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project. -->

## License
MIT 
