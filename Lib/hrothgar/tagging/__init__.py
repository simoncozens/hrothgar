"""Expressive tag prediction from a frozen shared glyph encoder.

This package predicts hand-labelled ``/Expressive/*`` tag scores (0-100 centiles)
for a font by pooling the frozen ``GlyphEncoder`` features of the rendered glyph
set into a *gap* vector (global mean pool, optionally plus the cross-glyph
variance) and feeding that through a small learned MLP head.

The encoder is frozen — it is read from a trained similarity checkpoint — so this
is a read-out of how much tag signal the shared encoder already carries, not a
jointly-trained model.  See ``tags.py`` (instance-aware loader), ``features.py``
(feature extraction), and ``train.py`` (the head training loop).
"""
