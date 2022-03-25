export plot_generated, plot_inference, plot_fe

using Plots
using PGFPlotsX
using Parameters
using LaTeXStrings
using ColorSchemes
using Colors
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");

function plot_generated(path, input, output, sig_len=length([input; output]), skip=10)
    plt_generated = @pgf GroupPlot(
    # group plot options
    {
        width="15cm", height="7cm",

        group_style = {
            group_size = "1 by 2",
            horizontal_sep = "0.5cm",
        },
    },

    # first
    {
    title="Generated NARMAX signal",
    ylabel="amplitude",
     legend_pos = "north west",
     legend_cell_align="{left}",
     grid = "major",
     legend_style = "{nodes={scale=1.0, transform shape}}",
     yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
     each_nth_point=skip,
     xtick_distance = 200,
     xmin=0.0, xmax=sig_len,
     xticklabels={},
    },
    Plot({no_marks, style={"ultra thick"}}, Coordinates(collect(1:sig_len), input[1:sig_len])), LegendEntry("input"),
    VLine({color="red"}, 1000),
    # second
    {

     xlabel=L"k",
     ylabel="amplitude",
     legend_pos = "north west",
     legend_cell_align="{left}",
     grid = "major",
     legend_style = "{nodes={scale=1.0, transform shape}}",
     yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
     each_nth_point=skip,
     xtick_distance = 200,
     xmin=0.0, xmax=sig_len,
    },
    Plot({no_marks, style={"ultra thick"}}, Coordinates(collect(1:sig_len), output[1:sig_len])), LegendEntry("output"),
    VLine({color="red"}, 1000),
    )

    pgfsave(path, plt_generated)
end


function plot_generated(path, input, output, sig_len=length([input; output]), skip=10)
    plt_generated = @pgf GroupPlot(
    # group plot options
    {
        width="15cm", height="7cm",

        group_style = {
            group_size = "1 by 2",
            horizontal_sep = "0.5cm",
        },
    },

    # first
    {
    title="Generated NARMAX signal",
    ylabel="amplitude",
     legend_pos = "north west",
     legend_cell_align="{left}",
     grid = "major",
     legend_style = "{nodes={scale=1.0, transform shape}}",
     yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
     each_nth_point=skip,
     xtick_distance = 200,
     xmin=0.0, xmax=sig_len,
     xticklabels={},
    },
    Plot({no_marks, style={"ultra thick"}}, Coordinates(collect(1:sig_len), input[1:sig_len])), LegendEntry("input"),
    VLine({color="red"}, 1000),
    # second
    {

     xlabel=L"k",
     ylabel="amplitude",
     legend_pos = "north west",
     legend_cell_align="{left}",
     grid = "major",
     legend_style = "{nodes={scale=1.0, transform shape}}",
     yticklabel_style={
        "/pgf/number format/fixed,
        /pgf/number format/precision=3"
        },
     each_nth_point=skip,
     xtick_distance = 200,
     xmin=0.0, xmax=sig_len,
    },
    Plot({no_marks, style={"ultra thick"}}, Coordinates(collect(1:sig_len), output[1:sig_len])), LegendEntry("output"),
    VLine({color="red"}, 1000),
    )

    pgfsave(path, plt_generated)
end

