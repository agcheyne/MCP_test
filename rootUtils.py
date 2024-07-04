import ROOT

# Create a ROOT histogram from a numpy array
def create_TH1D_from_np_array(array, name, title, nbins, xmin, xmax):
    h = ROOT.TH1D(name, title, nbins, xmin, xmax)
    for x in array:
        h.Fill(x)
    return h

# Create a ROOT histogram from a pandas dataframe
def create_TH1D_from_pandas_df(df, column, name, title, nbins, xmin, xmax):
    h = ROOT.TH1D(name, title, nbins, xmin, xmax)
    for x in df[column]:
        h.Fill(x)
    return h

# Create a 2d ROOT histogram from a pandas dataframe

def create_TH2D_from_pandas_df(df, column1, column2, name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax):
    h = ROOT.TH2D(name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax)
    for x, y in zip(df[column1], df[column2]):
        h.Fill(x, y)
    return h


# Create a ROOT TGraph from a pandas dataframe
def create_TGraph_from_pandas_df(df, column1, column2, name, title):
    g = ROOT.TGraph(df.shape[0])
    for i, (x, y) in enumerate(zip(df[column1], df[column2])):
        g.SetPoint(i, x, y)
    return g

# Create a ROOT TGraphErrors from a pandas dataframe
def create_TGraphErrors_from_pandas_df(df, column1, column2, column3, column4, name, title):
    g = ROOT.TGraphErrors(df.shape[0])
    for i, (x, y, ex, ey) in enumerate(zip(df[column1], df[column2], df[column3], df[column4])):
        g.SetPoint(i, x, y)
        g.SetPointError(i, ex, ey)
    return g

# Create a ROOT TGraphAsymmErrors from a pandas dataframe
def create_TGraphAsymmErrors_from_pandas_df(df, column1, column2, column3, column4, column5, column6, name, title):
    g = ROOT.TGraphAsymmErrors(df.shape[0])
    for i, (x, y, exl, exh, eyl, eyh) in enumerate(zip(df[column1], df[column2], df[column3], df[column4], df[column5], df[column6])):
        g.SetPoint(i, x, y)
        g.SetPointError(i, exl, exh, eyl, eyh)
    return g

# Create a ROOT TGraph2D from a pandas dataframe
def create_TGraph2D_from_pandas_df(df, column1, column2, column3, name, title):
    g = ROOT.TGraph2D(df.shape[0])
    for i, (x, y, z) in enumerate(zip(df[column1], df[column2], df[column3])):
        g.SetPoint(i, x, y, z)
    return g

# Create a ROOT TGraph2DErrors from a pandas dataframe
def create_TGraph2DErrors_from_pandas_df(df, column1, column2, column3, column4, column5, column6, name, title):
    g = ROOT.TGraph2DErrors(df.shape[0])
    for i, (x, y, z, ex, ey, ez) in enumerate(zip(df[column1], df[column2], df[column3], df[column4], df[column5], df[column6])):
        g.SetPoint(i, x, y, z)
        g.SetPointError(i, ex, ey, ez)
    return g


# Create a ROOT TProfile from a pandas dataframe
def create_TProfile_from_pandas_df(df, column1, column2, name, title, nbinsx, xmin, xmax, ymin, ymax):
    p = ROOT.TProfile(name, title, nbinsx, xmin, xmax, ymin, ymax)
    for x, y in zip(df[column1], df[column2]):
        p.Fill(x, y)
    return p

# Create a ROOT TProfile2D from a pandas dataframe
def create_TProfile2D_from_pandas_df(df, column1, column2, column3, name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax, zmin, zmax):
    p = ROOT.TProfile2D(name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax, zmin, zmax)
    for x, y, z in zip(df[column1], df[column2], df[column3]):
        p.Fill(x, y, z)
    return p

# Create a ROOT THStack from a list of histograms
def create_THStack(h_list, name, title):
    hs = ROOT.THStack(name, title)
    for h in h_list:
        hs.Add(h)
    return hs

# Create a ROOT TLegend from a list of histograms
def create_TLegend(h_list, name, title):
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    for h in h_list:
        leg.AddEntry(h, h.GetTitle(), "l")
    return leg

# Create a ROOT TCanvas with a list of histograms
def create_TCanvas_with_histos(h_list, name, title):
    c = ROOT.TCanvas(name, title, 800, 600)
    for i, h in enumerate(h_list):
        if i == 0:
            h.Draw()
        else:
            h.Draw("same")
    return c
