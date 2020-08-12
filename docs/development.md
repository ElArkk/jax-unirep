# Development

This page will show you the officially-supported ways
of getting set up with a development environment
so that you can hack on jax-unirep and make contributions!

## Get familiar with community practices

Kevin Markham [has a great resource](https://www.dataschool.io/how-to-contribute-on-github/)
on how to contribute to open source software
on GitHub.
We'd really encourage you to look through it first
if you're not already familiar with canonical, community workflow practices
that have been adopted across multiple open source projects.

The most basic ideas that you'll need to grasp are:

1. Making forks.
2. Local vs. remote.

## VSCode Dev Containers

The easiest way for you to get setup is to use a dev container with VSCode.
(We're not paid by Microsoft, we're just fans of this way of working.)

To get started:

1. Fork the repository.
1. Ensure you have Docker running on your local machine.
1. Ensure you have VSCode running on your local machine.
1. In Visual Studio Code,
   click on the quick actions Status Bar item in the lower left corner.
1. Then select “Remote Containers: Open Repository In Container”.
1. Enter in the URL of your fork of pyjanitor.

VSCode will pull down the prebuilt Docker container,
git clone the repository for you inside an isolated Docker volume,
and mount the repository directory inside your Docker container.

Follow best practices to submit a pull request by making a feature branch.
Now, hack away, and submit in your pull request!

You shouln’t be able to access the cloned repo on your local hard drive.
If you do want local access,
then clone the repo locally first before selecting
“Remote Containers: Open Folder In Container”.

If you find something is broken because a utility is missing in the container,
submit a PR with the appropriate build command inserted in the Dockerfile.
Care has been taken to document what each step does,
so please read the in-line documentation in the Dockerfile carefully.

## Conda Environment

This is another supported way of working.
We assume that you already have the Anaconda distribution of Python
setup on your local machine.
Once you've done that:

1. Fork the repository.
2. Clone your fork locally.
3. In your terminal, enter into the local copy of the repository.

Now, install the environment:

```bash
conda env create -f environment.yml
```

This will create an environment called `jax-unirep` that you can activate.

```bash
conda activate jax-unirep
```

Finally, install `jax-unirep` into your environment in development mode.

```bash
python setup.py develop
```

## Your favourite way here

If you've got well-documented steps for how to get setup,
come contribute them as part of the docs here!
You'll want to edit `docs/development.md`.
Then submit a pull request in: everyone in the community will benefit!
