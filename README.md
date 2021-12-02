# ei-mcoe-map
Compile data to generate an interactive map of fossil generators and their marginal cost
of electricity

You will need to obtain API keys from FRED and EIA in order to access the inflation data
and monthly historical fuel costs by state, which are used in the analysis.

* [Request an EIA API key](https://www.eia.gov/opendata/register.php)
* [Request a FRED API key](https://fredaccount.stlouisfed.org/login/secure/)

Store these API keys in environment variables named `API_KEY_EIA` and `API_KEY_FRED` so
that the software has access to them. You can also set the environment variables within
the "Setup" section of the Jupyter notebook.

Note: the MCOE output uses EIA NEMS data to estimate the split between fixed and
variable O&M costs. This data is stored in the `inputs` directory. We've only extracted
the 2019 NEMS data, so 2019 is currently being used to estimate the 2020 O&M costs.
Which year the 2019 data is associated with is determined by `NEMS_YEAR` in
`ei_mcoe.py`.

## Installation
To install the software in this repository, clone it to your computer using git. If
you're authenticating using SSH:
```sh
git clone git@github.com:catalyst-cooperative/ei-mcoe-map.git
```
Or if you're authenticating via HTTPS:
```sh
git clone https://github.com/catalyst-cooperative/ei-mcoe-map.git
```

Then in the top level directory of the repository, create a `conda` environment based on
the `environment.yml` file that is stored in the repo:

```sh
conda env create --file environment.yml
```

Note that the software in this repository depends on [the dev
branch](https://github.com/catalyst-cooperative/pudl/tree/dev) of the [main PUDL
repository](https://github.com/catalyst-cooperative/pudl), and the `setup.py` in this
repository indicates that it should be installed directly from GitHub. This can be a bit
slow, as `pip` (which in this case is running inside of a `conda` environment) clones
the entire history of the repository containing the package being installed. How long it
takes will depend on the speed of your network connection. It might take ~5 minutes.

The `environment.yml` file also specifies that the Python package defined within this
repository should be installed such that it is editable.  This will allow you to change
the modules that are part of the repository and have the installed software reflect your
changes.

If you want to make changes to the PUDL software as well, you can clone the PUDL
repository into another directory (outside of this repository), and direct `conda` to
install the package from there. A commented out example of how to do this is included
in `environment.yml`. **NOTE:** if you want to install PUDL in editable mode from a
locally cloned repo, you'll need to comment out the dependency in `setup.py` as it may
otherwise conflict with the local installation (pip can't resolve the precedence of
different git based versions).

After any changes to the environment specification, you'll need to recreate the conda
environment. The most reliable way to do that is to remove the old environment and
create it from scratch. If you're in the top level `ei-mcoe-map` directory and have
the `ei-mcoe` environment activated, that process would look like this:

```sh
conda deactivate
conda env remove --name ei-mcoe
conda env create --file environment.yml
conda activate ei-mcoe
```

In order to use this repository, you will need a recent copy of the PUDL database. You
You can either create one for yourself by [running the ETL
pipeline](https://catalystcoop-pudl.readthedocs.io/en/latest/dev/run_the_etl.html), or
you can follow the instructions in the [PUDL examples
repository](https://github.com/catalyst-cooperative/pudl-examples) to download the
pre-processed data alongside a Docker container.

To work with the pre-processed data **outside** of the Docker container, you will need
to tell the PUDL software where to find that data on your computer. When you extract the
pre-processed data archive, it will include a directory named `pudl_data` -- you need to
put the path to that directory in a file called `.pudl.yml` in your home directory. The
contents will need to look like the following (but with real paths...):

```yml
pudl_in: /path/to/your/downloaded/pudl_data
pudl_out: /the/same/path/to/pudl_data
```
