import ROOT
import os


def set_plot_style(histograms, title, nDivision):

    for hist in histograms:
        hist.SetStats(0)
        hist.GetXaxis().SetTitle(title)
        hist.GetXaxis().CenterTitle()
        hist.GetXaxis().SetTitleOffset(1.3)
        hist.GetXaxis().SetTitleSize(0.05)
        hist.GetXaxis().SetNdivisions(nDivision)


def fill_histograms(data, variable_name, bin_size, lower_limit, upper_limit):

    total_tracks = ROOT.TH1D('', '', bin_size, lower_limit, upper_limit)
    ghost = ROOT.TH1D('', '', bin_size, lower_limit, upper_limit)

    for index, row in data.iterrows():
        total_tracks.Fill(row[variable_name])  # Fill for all tracks (downstream and ghost)
    
    # Fill ghost histogram only when the track is not matched
        if not row['isMatched']:
            ghost.Fill(row[variable_name])


    return total_tracks, ghost



def comparePlots(data_baseline, data_seed_track, title, variable_name, lower_limit, upper_limit, nDivision, bin_type, plot_path,plot_category, canvas_width=1200, canvas_height=800):
    canvas = ROOT.TCanvas('name', title, canvas_width, canvas_height)
    canvas.SetRightMargin(0.09)
    canvas.SetLeftMargin(0.09)
    canvas.SetBottomMargin(0.15)

    legend = ROOT.TLegend(0.67, 0.65, 0.88, 0.85) 

    text = ROOT.TPaveText(0.75, 0.2, 0.899, 0.23, "NDC")

    bin_size = 300 if bin_type == 'BinType1' else 15

    # Fill histograms for both datasets
    total_tracks_baseline, ghost_baseline = fill_histograms(data_baseline, variable_name, bin_size, lower_limit, upper_limit)
    total_tracks_seed_track, ghost_seed_track = fill_histograms(data_seed_track, variable_name, bin_size, lower_limit, upper_limit)


    histograms = [total_tracks_baseline, total_tracks_seed_track, ghost_baseline, ghost_seed_track]
    set_plot_style(histograms, title, nDivision)

    # Set fill colors and alpha
    # for hist, fill_color, alpha in zip(histograms, [46, 45, 38, 40], [0.8, 0.4, 0.9, 0.4]):
    if plot_category=='Baseline':
        for hist, fill_color, alpha in zip(histograms, [38, 38, 46, 46], [0.4, 0.9, 0.4, 0.9]):    
            hist.SetFillColorAlpha(fill_color, alpha)
    elif plot_category=='Final':
        for hist, fill_color, alpha in zip(histograms, [38, 38, 46, 46], [0.4, 0.9, 0.4, 0.9]):    
            hist.SetFillColorAlpha(fill_color, alpha)



    # Draw the histograms
    total_tracks_baseline.Draw('HIST')
    total_tracks_seed_track.Draw('HIST SAME')
    ghost_baseline.Draw('HIST SAME')
    ghost_seed_track.Draw('HIST SAME')

    text.AddText('LHCb Simulation')
    text.SetBorderSize(1)

    # Add entries to legend
    legend.AddEntry(total_tracks_baseline, 'Baseline - Total Tracks', 'f')
    legend.AddEntry(total_tracks_seed_track, 'FinalModel - Total Tracks', 'f')
    legend.AddEntry(ghost_baseline, 'Baseline - Ghost Tracks', 'f')
    legend.AddEntry(ghost_seed_track, 'FinalModel - Ghost Tracks', 'f')
    
    # legend.SetTextSize(0.035)
    legend.SetMargin(0.2)  # Increase margin inside the legend box
    # legend.SetEntrySeparation(0.6)

    legend.Draw()
    text.Draw()
    canvas.Draw()

    os.makedirs(plot_path, exist_ok=True)
    return canvas.SaveAs(f"{plot_path}/{title}.png")
