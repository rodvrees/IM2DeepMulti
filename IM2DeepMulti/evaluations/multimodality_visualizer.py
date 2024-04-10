import seaborn as sns
import sys
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import alphatims.utils
import alphatims.bruker
import alphatims.plotting

plt.rcParams["axes.facecolor"] = "black"


def im2ccs(reverse_im, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    Convert ion mobility to collisional cross section.

    Parameters
    ----------
    reverse_im
        Reduced ion mobility.
    mz
        Precursor m/z.
    charge
        Precursor charge.
    mass_gas
        Mass of gas, default 28.013
    temp
        Temperature in Celsius, default 31.85
    t_diff
        Factor to convert Celsius to Kelvin, default 273.15

    Notes
    -----
    Adapted from theGreatHerrLebert/ionmob (https://doi.org/10.1093/bioinformatics/btad486)

    """

    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (
        np.sqrt(reduced_mass * (temp + t_diff)) * 1 / reverse_im
    )

def ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert CCS to 1 over reduced ion mobility (1/k0)
    :param ccs: collision cross-section
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas (N2)
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return  ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (SUMMARY_CONSTANT * charge)

def plot_peptide_mobilogram(
    data,
    observed_mobilities,
    predicted_ccs,
    peptide,
    ppm=50,
    rt_tolerance=30,
    path="/home/robbe/IM2DeepMulti/figs/mobilograms/",
):
    """Plot the mobilogram of a peptide. As well as the heatmap of the peptide."""
    precursor_mz = peptide["mz"]
    precursor_rt = peptide["rt"]
    rt_slice = slice(precursor_rt - rt_tolerance, precursor_rt + rt_tolerance)
    precursor_mz_slice = slice(
        precursor_mz / (1 + ppm / 10**6), precursor_mz * (1 + ppm / 10**6)
    )
    precursor_indices = data[
        rt_slice,
        :,
        0,  # index 0 means that the quadrupole is not used
        precursor_mz_slice,
        "raw",
    ]
    df = data.as_dataframe(precursor_indices)
    if len(df) > 100:
        # Filter out 10 % least intense points
        df = df[df["intensity_values"] > df["intensity_values"].quantile(0.5)]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    sns.scatterplot(
        data=df,
        x="rt_values",
        y="mobility_values",
        hue="intensity_values",
        palette=list(reversed(cc.fire)),
        legend=False,
        ax=axes[0],
    )
    sns.lineplot(data=df, x="mobility_values", y="intensity_values", ax=axes[1])
    for i in observed_mobilities:
        axes[1].axvline(
            x=i,
            color="r",
            linestyle="--",
            alpha=0.3,
        )
        axes[1].text(
            x=i,
            y=df["intensity_values"].mean(),
            s=round(im2ccs(i, peptide["mz"], peptide["charge"]), 0),
            c="w",
        )
        axes[0].axhline(y=i, color="w", linestyle="--", alpha=0.8)
        axes[0].text(
            x=df["mobility_values"].mean(),
            y=i,
            s=round(im2ccs(i, peptide["mz"], peptide["charge"]), 0),
            c="w",
        )
    for i in predicted_ccs:
        axes[1].axvline(x=ccs_to_one_over_reduced_mobility(i), color="c", linestyle="--", alpha=0.3)
        axes[1].text(x=i, y=df["intensity_values"].mean(), s=round(i, 0), c="r")
        axes[0].axhline(y=ccs_to_one_over_reduced_mobility(i), color="c", linestyle="--", alpha=0.8)
        axes[0].text(x=df["mobility_values"].mean(), y=i, s=round(i, 0), c="r")
    axes[0].set_title("RT-IM Heatmap")
    axes[1].set_title("Peptide ion mobilogram")
    axes[0].set_xlabel("Retention time (s)")
    axes[0].set_ylabel("Ion mobility (1/K0)")
    axes[1].set_ylabel("Intensity")
    axes[1].set_xlabel("Ion mobility (1/K0)")
    fig.suptitle(f'{peptide["sequence"]} {peptide["charge"]}')
    fig.savefig(path + f'{peptide["sequence"]}_{peptide["charge"]}.png')


def extract_peptide_info(row):
    """Extract the peptide information from the evidence file."""
    peptide = {
        "sequence": row["Modified sequence"],
        "charge": row["charge"],
        "mz": row["m/z"],
        "rt": row["Retention time"] * 60,
    }
    return peptide


def get_mobilities(data, peptide):
    """Get the observed mobilities of a peptide."""
    peptide_data = data[
        (data["Modified sequence"] == peptide["sequence"])
        & (data["Charge"] == peptide["charge"])
    ]
    observed_mobilities = peptide_data["1/K0"].values
    return list(observed_mobilities)


def get_predicted_ccs(data, peptide):
    """Get the predicted CCS of a peptide."""
    peptide_data = data[
        (data["Modified sequence"] == peptide["sequence"])
        & (data["Charge"] == peptide["charge"])
    ]
    predicted_ccs = peptide_data["Predicted CCS"].values
    return list(predicted_ccs)


def main():
    print("Starting")
    # Load the data
    dataset = pd.read_csv(
        "/home/robbe/IM2DeepMulti/dataset/QD055SCB1_C3_Slot2-41_1_16334_predictions.csv"
    )
    raw_data = alphatims.bruker.TimsTOF(
        "/home/robbe/IM2DeepMulti/dataset/evaluation_raw_data/QD055SCB1_C3_Slot2-41_1_16334.d"
    )
    print("Data loaded")

    # Make a plot for each peptidoform
    for i, row in dataset.iterrows():
        peptide = extract_peptide_info(row)
        print(peptide)
        observed_mobilities = get_mobilities(dataset, peptide)
        predicted_ccs = get_predicted_ccs(dataset, peptide)
        print(observed_mobilities)
        plot_peptide_mobilogram(
            raw_data,
            observed_mobilities,
            predicted_ccs,
            peptide,
            ppm=10,
            rt_tolerance=10,
            path="/home/robbe/IM2DeepMulti/figs/mobilograms/QD055SCB1_C2_Slot2-41_1_16334",
        )


if __name__ == "__main__":
    main()
