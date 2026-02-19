"""Vocal tract simulator: research-grade articulatory synthesis engine.

Physical model of the human vocal tract based on published research:
- Glottal source: Fant, Liljencrants & Lin (1985) LF model
- Vocal tract: Kelly & Lochbaum (1962) waveguide
- Articulators: Maeda (1990) 7-parameter PCA model
- Area function: PCA basis vectors → cross-sectional areas
- Nasal tract: Dang & Honda (1996) fixed-geometry side-branch
- Radiation: Flanagan (1972) lip radiation filter
- Coarticulation: Anticipatory + carryover blending
- Phonation: f0 contours, voice quality dynamics
- Subglottal: Respiratory pressure model
- Singing: Vibrato, registers, singer's formant
- Expression: Emotion → vocal tract configuration
"""
