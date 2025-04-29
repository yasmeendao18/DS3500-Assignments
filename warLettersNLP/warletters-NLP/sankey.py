"""
Lilian Uong and Yasmeen Dao
DS3500 / warletters-NLP
A reusable extensible framework for Natural Language Processing  / Homework 3
2/23/23 / 2/28/23
"""

import plotly.graph_objects as go

def _code_mapping(df, src, targ):
    # Get distinct labels
    labels = sorted(list(set(list(df[src]) + list(df[targ]))))

    # Get integer codes
    codes = list(range(len(labels)))

    # Create label to code mapping
    lc_map = dict(zip(labels, codes))

    # Substitute names for codes in dataframe
    df = df.replace({src: lc_map, targ: lc_map})

    return df, labels

def make_sankey(df, src, targ, *cols, vals=None, **kwargs):
    """ Create a sankey diagram linking src values to
    target values with thickness vals """

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    df, labels = _code_mapping(df, src, targ)
    link = {'source':df[src], 'target':df[targ], 'value':values}
    pad = kwargs.get('pad', 15)

    node = {'label': labels, 'pad': pad}
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    fig.show()