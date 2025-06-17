# anodet

A set of functions and classes for performing anomaly detection in images using features from pretrained neural networks.

The package includes functions and classes for extracting, modifying and comparing features. It also includes unofficial implementations of [**PaDiM**](https://arxiv.org/abs/2011.08785) and [**PatchCore**](https://arxiv.org/abs/2106.08265).

Some code has been borrowed and/or inspired by other repositories, see code reference below.

See [wiki](https://github.com/OpenAOI/anodet/wiki) for documentation.

#### Example result with padim on image from [MVTEC dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
![](notebooks/example_images/padim_example_image.png)

## Installation

Clone the repository
```
git clone https://github.com/OpenAOI/anodet.git
```

Install the package

```
cd anodet
python -m pip install -r requirements.txt
python -m pip install .
```


## Usage example

```
# Prepare a dataloader and fit a model to the data
dataloader = DataLoader(...)
padim = anodet.Padim() 
padim.fit(dataloader)

# Prepare some test images and make predictions
batch = ...
image_scores, score_map = padim.predict(batch) 
```

See [notebooks](https://github.com/OpenAOI/anodet/tree/master/notebooks) for in depth examples.


## Development setup

#### Install

Install the package in editable mode
```
python -m pip install --editable [PATH TO REPOSITORY]
```

#### Tests

Install packages for testing
```
python -m pip install pytest pytest-mypy pytest-flake8
```

Run tests
```
cd [PATH TO REPOSITORY]
pytest --mypy --flake8
```

For configuration of pytest, mypy and flake8 edit `setup.cfg`.


#### Creating docs

Install pydoc-markdown
```
python -m pip install pydoc-markdown
```

Clone docs repository
```
git clone https://github.com/OpenAOI/anodet.wiki.git
```

Run script
```
cd anodet.wiki
python generate_docs.py --source-path=[PATH TO REPOSITORY] --package-name="anodet" --save-path=.
```




## Code Reference

PaDiM:
https://arxiv.org/abs/2011.08785

PatchCore:
https://arxiv.org/abs/2106.08265

Some parts used in patch_core.py :
https://github.com/hcw-00/PatchCore_anomaly_detection

Code in directory sampling_methods :
https://github.com/google/active-learning

concatenate_two_layers function in feature_extraction.py :
https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

pytorch_cov function in utils.py :
https://github.com/pytorch/pytorch/issues/19037




Absolutely! Here's a **revamped, very catchy, simplified, and motivating script** for each slide. It's designed to keep even beginners hooked and excited to finish the entire MLOps setup.

---

### 🎬 **Slide 1: Title – Industrial MLOps Stack Setup Guide**

🎙️
"Welcome! 🚀
You're about to learn how to build a **professional MLOps environment** on Windows—yes, Windows, no Linux, no WSL!
We’re going to combine **cloud tools, containers, automation, and Python** into one powerful system.
By the end, you won’t just understand MLOps—you’ll be ready to use it at work or in real projects.
Let’s dive in and start this journey!"

---

### 🎯 **Slide 2: What You'll Build Today**

🎙️
"Today, you're not just installing tools.
You’re building a **complete enterprise-level machine learning workflow**—the kind used by real companies.
We’ll set up infrastructure, tools, and automation that work together smoothly.
By the time we’re done, you’ll have your own **MLOps factory**—running right on your laptop.
Ready to build something awesome? Let’s go!"

---

### 🏗️ **Slide 3: Architecture Overview**

🎙️
"Here’s the blueprint of your future MLOps stack.
Think of it like building a smart factory for machine learning.
We have 3 layers:

1. **Development** – Code, Python, version control
2. **Containers** – Docker, Kubernetes
3. **Cloud** – Azure, CI/CD, monitoring, and more
   When they all connect, magic happens. And you’ll build it all, step-by-step!"

---

### ⚠️ **Slide 4: Prerequisites**

🎙️
"Before we build, let’s make sure your system is ready.
⚡ Windows Pro or Enterprise
💾 At least 16GB RAM
🔐 Admin access
This avoids slowdowns and setup errors.
Trust me, getting this part right makes the rest easy—and fun!"

---

### ☁️ **Slide 5: Azure CLI Installation**

🎙️
"Now we’re talking cloud. ☁️
Azure CLI is your command center to talk with the cloud.
With just a few commands, you’ll control compute, storage, and machine learning resources from your own terminal.
It’s fast, flexible, and free to get started—let’s set it up now!"

---

### 🐳 **Slide 6: Docker Desktop Setup**

🎙️
"This is HUGE.
Docker lets you package your ML code into **containers**—portable, repeatable environments.
No more “it works on my machine” problems.
We’ll also turn on **Kubernetes** so you can scale things like a pro.
It’s easy—we’ll walk through it together."

---

### 🐍 **Slide 7: Python Installation**

🎙️
"You already know Python. But managing versions? That’s tricky.
We’ll install Python **the smart way** using pyenv—so you can run multiple projects without conflicts.
It’s clean, professional, and totally worth it.
This step alone will save you hours in the future!"

---

### 📦 **Slide 8: Poetry - Dependency Management**

🎙️
"Let’s manage your Python packages like a pro.
Poetry makes sure your environment is clean, your dependencies are locked, and your projects are easy to share or deploy.
It’s like pip + virtualenv + packaging all in one—but smarter.
You’ll love it. Let’s get it set up."

---

### 🔧 **Slide 9: MLOps Libraries Installation**

🎙️
"Time to power up!
We’ll install the essential MLOps tools:
✅ Azure ML SDK – connect to the cloud
✅ MLflow – track experiments
✅ Key Vault & Monitoring – for real production work
✅ Plus, all your ML libraries: pandas, scikit-learn, plotly…
This is your ultimate ML toolbox—ready to go."

---

### ✅ **Slide 10: Verification Checklist**

🎙️
"Quick checkpoint—let’s make sure everything’s working before we move on.
You’ll test Docker, Python, Azure CLI, MLflow, and more.
If everything’s green, you’re golden.
You now have a **full MLOps stack running locally**, just like the pros."

---

### 🎉 **Slide 11: Next Steps Overview**

🎙️
"Look at what you’ve built! 🎉
From scratch to full-stack MLOps in under an hour.
What’s next?

1. Build real ML pipelines
2. Automate deployments
3. Monitor models in production
   This is your springboard into **real-world AI engineering**. You’ve got this!"

---

### 🙏 **Slide 12: Thank You**

🎙️
"You made it!
You now have a complete, working MLOps setup—and the skills to grow even further.
I’m proud of you. This is where real learning begins.
Now go build. Go explore. Go lead.
And remember—**you’re MLOps-ready.** 💪
Thanks for watching. More awesome tutorials coming soon!"

---

Would you like this as a downloadable script document, or should I help break it into subtitles or narration blocks?
