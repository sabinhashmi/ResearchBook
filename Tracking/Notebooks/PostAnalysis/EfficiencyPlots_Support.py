import ROOT

def get_histograms(root_file, variable_prefix, line_color, marker_color, fill_color, plot_title):
    variable_total = root_file.Get(f"{variable_prefix}_Total")
    variable_ghosts = root_file.Get(f"{variable_prefix}_Ghosts")
    
    variable_true = variable_total.Clone()
    variable_true.Add(variable_ghosts, -1) 

    category_list = [
    "01_UT+T_",
    "02_UT+T_P>5GeV_",
    "03_UT+T_strange_",
    "04_UT+T_strange_P>5GeV_",
    "05_noVelo+UT+T_strange_",
    "06_noVelo+UT+T_strange_P>5GeV_",
    "07_UT+T_fromDB_",
    "08_UT+T_fromBD_P>5GeV_",
    "09_noVelo+UT+T_fromBD_",
    "10_noVelo+UT+T_fromBD_P>5GeV_",
    "11_UT+T_SfromDB_",
    "12_UT+T_SfromDB_P>5GeV_",
    "13_noVelo+UT+T_SfromDB_",
    "14_noVelo+UT+T_SfromDB_P>5GeV_"
]




    # DownstreamTracks
    # reconstructed = root_file.Get(f'05_noVelo+UT+T_strange_{variable_prefix}_reconstructed')
    # reconstructible = root_file.Get(f'05_noVelo+UT+T_strange_{variable_prefix}_reconstructible')



    total_reconstructed = None
    total_reconstructible = None

    for category in category_list:
        reconstructed_ = root_file.Get(f'{category}{variable_prefix}_reconstructed')
        reconstructible_ = root_file.Get(f'{category}{variable_prefix}_reconstructible')

        if total_reconstructed is None:
            total_reconstructed = reconstructed_.Clone("total_reconstructed")
            total_reconstructible = reconstructible_.Clone("total_reconstructible")
        else:
            total_reconstructed.Add(reconstructed_)
            total_reconstructible.Add(reconstructible_)

        


    efficiency = total_reconstructed.Clone()
    efficiency.Divide(total_reconstructible)

    ghost_ratio = variable_ghosts.Clone()
    ghost_ratio.Divide(variable_total)

   
    efficiency.SetMarkerStyle(20)
    efficiency.SetMarkerSize(0.8)
    efficiency.SetLineColor(ROOT.kBlack)
    efficiency.SetMarkerColor(marker_color)

    ghost_ratio.SetMarkerStyle(20)
    ghost_ratio.SetMarkerSize(0.8)
    ghost_ratio.SetLineColor(ROOT.kBlack)
    ghost_ratio.SetMarkerColor(marker_color)

    variable_true.SetLineColor(line_color)  # Line color for variable
    variable_true.SetFillColorAlpha(fill_color, 0.5)

    variable_true.SetStats(0)
    variable_true.SetTitle(plot_title)

    return variable_true, efficiency, ghost_ratio

# Helper function to create legends
def create_legend(entries, position=(0.7, 0.2, 0.9, 0.25)):
    legend = ROOT.TLegend(*position)
    for hist, label, option in entries:
        legend.AddEntry(hist, label, option)
    return legend



def canvas_plot(baseline, upgrade, kinematics, title,plot_type,save_path,x_limit=None):

    # Set up the canvas
    canvas = ROOT.TCanvas("", "", 1500, 600)
    canvas.SetRightMargin(0.08)
    canvas.SetLeftMargin(0.08)
    canvas.SetBottomMargin(0.1)


    # Retrieve baseline histograms and set styles (rebinning applied)
    base_variable, base_efficiency, base_ghost_ratio = get_histograms(
        baseline, kinematics, ROOT.kBlue, ROOT.kBlue, 38, plot_title=title, 
        )

    # Retrieve seed histograms and set styles (rebinning applied)
    seed_variable, seed_efficiency, seed_ghost_ratio = get_histograms(
        upgrade, kinematics, ROOT.kRed, ROOT.kRed, 45, plot_title=title, 
        )
    

    if x_limit:
        base_variable.GetXaxis().SetRangeUser(x_limit[0], x_limit[1])
        seed_variable.GetXaxis().SetRangeUser(x_limit[0], x_limit[1])
        base_efficiency.GetXaxis().SetRangeUser(x_limit[0], x_limit[1])
        seed_efficiency.GetXaxis().SetRangeUser(x_limit[0], x_limit[1])
        base_ghost_ratio.GetXaxis().SetRangeUser(x_limit[0], x_limit[1])
        seed_ghost_ratio.GetXaxis().SetRangeUser(x_limit[0], x_limit[1])



    max_variable = base_variable.GetMaximum()
    if max_variable > 0:
        base_variable.Scale(1.0 / max_variable)
        seed_variable.Scale(1.0 / max_variable)


    max_efficiency = base_efficiency.GetMaximum()
    if max_efficiency > 0:
        base_efficiency.Scale(1.0 / max_efficiency)
        seed_efficiency.Scale(1.0 / max_efficiency)
    
    max_ghost_ratio = base_ghost_ratio.GetMaximum()
    if max_ghost_ratio > 0:
        base_ghost_ratio.Scale(1.0 / max_ghost_ratio)
        seed_ghost_ratio.Scale(1.0 / max_ghost_ratio)
   



    
    # Create legends
    legend = create_legend(
        [
            (base_variable, "True Downstream Tracks", "f"),
            (base_efficiency, "Baseline-Efficiency", "lep"),
            (seed_efficiency, "Upgrade-Efficiency", "lep")
        ],
        (0.8, 0.15, 0.9, 0.25),
            )

    if plot_type=='Efficiency':

        base_variable.Draw("HIST")
        base_efficiency.Draw("E1 SAME")
        seed_efficiency.Draw("E1 SAME")

    
    elif plot_type=='GhostRatio':
        base_variable.Draw("HIST")
        base_ghost_ratio.Draw("E1 SAME")
        seed_ghost_ratio.Draw("E1 SAME")

    else:
        raise ValueError("Invalid plot type. Choose between 'Efficiency' and 'GhostRatio'.")
    
    legend.Draw()
    canvas.Draw()
    canvas.SaveAs(f"{save_path}/{title}.png")



