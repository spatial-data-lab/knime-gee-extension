"""
Built-in color palettes for GEE Image / Image Collection View nodes.

Matplotlib / ColorBrewer entries have no prefix (e.g. ``Viridis``, ``RdBu``).
Other libraries use ``library:name`` (e.g. ``cmocean:balance``). ``Custom`` is first.
"""

# Custom must stay first (empty = use Color palette parameter).
_MATPLOTLIB_PRESETS = {
    "Blues": "EFF3FF,C6DBEF,9ECAE1,6BAED6,4292C6,2171B5,084594",
    "BrBG": "8C510A,D8B365,F6E8C3,F5F5F5,C7EAE5,5AB4AC,01665E",
    "Bwr": "blue,white,red",
    "Cividis": "00204C,213D6B,555B6C,7B7A77,A59C74,D3C064,FFE945",
    "Coolwarm": "3B4CC0,6F91F2,A9C5FC,DDDDDD,F6B69B,E6745B,B40426",
    "Greens": "EDF8E9,C7E9C0,A1D99B,74C476,41AB5D,238B45,005A32",
    "Greys": "F7F7F7,D9D9D9,BDBDBD,969696,737373,525252,252525",
    "Greys_r": "252525,525252,737373,969696,BDBDBD,D9D9D9,F7F7F7",
    "Inferno": "000004,320A5A,781B6C,BB3654,EC6824,FBB41A,FCFFA4",
    "Magma": "000004,2C105C,711F81,B63679,EE605E,FDAE78,FCFDBF",
    "Plasma": "0D0887,5B02A3,9A179B,CB4678,EB7852,FBB32F,F0F921",
    "PRGn": "762A83,AF8DC3,E7D4E8,F7F7F7,D9F0D3,7FBF7B,1B7837",
    "PuBuGn": "F6EFF7,D0D1E6,A6BDDB,67A9CF,3690C0,02818A,016450",
    "PuOr": "B35806,F1A340,FEE0B6,F7F7F7,D8DAEB,998EC3,542788",
    "RdBu": "B2182B,EF8A62,FDDBC7,F7F7F7,D1E5F0,67A9CF,2166AC",
    "RdGy": "B2182B,EF8A62,FDDBC7,FFFFFF,E0E0E0,999999,4D4D4D",
    "RdYlBu": "D73027,FC8D59,FEE090,FFFFBF,E0F3F8,91BFDB,4575B4",
    "RdYlGn": "D73027,FC8D59,FEE090,FFFFBF,D9EF8B,A6D96A,1A9850",
    "Reds": "FEE5D9,FCBBA1,FC9272,FB6A4A,EF3B2C,CB181D,99000D",
    "Spectral": "D53E4F,FC8D59,FEE08B,FFFFBF,E6F598,99D594,3288BD",
    "Viridis": "440154,433982,30678D,218F8B,36B677,8ED542,FDE725",
    "YlOrRd": "FFFFB2,FED976,FEB24C,FD8D3C,FC4E2A,E31A1C,B10026",
}

_PALETTE_BY_GROUP = {
    "colorcet": {
        "CET_C2": "EF55F2,FCC882,B8E014,32AD26,2F5DB9,712AF7,ED53F3",
    },
    "cmocean": {
        "algae": "D7F9D0,A2D595,64B463,129450,126E45,1A482F,122414",
        "balance": "181C43,0C5EBE,75AABE,F1ECEB,D08B73,A52125,3C0912",
        "deep": "FDFECC,A5DFA7,5DBAA4,488E9E,3E6495,3F396C,281A2C",
        "delta": "112040,1C67A0,6DB6B3,FFFCCC,ABAC21,177228,172313",
        "gray": "000000,232323,4A4A49,727171,9B9A9A,CACAC9,FFFFFD",
        "haline": "2A186C,14439C,206E8B,3C9387,5AB978,AAD85C,FDEF9A",
        "ice": "040613,292851,3F4B96,427BB7,61A8C7,9CD4DA,EAFDFD",
        "solar": "331418,682325,973B1C,B66413,CB921A,DAC62F,E1FD4B",
        "thermal": "042333,2C3395,744992,B15F82,EB7958,FBB43D,E8FA5B",
    },
    "crameri": {
        "batlow": "011959,0E365E,1D5561,3E6C55,687B3E,9B882E,D59448,F9A380,FDB7BD,FACCFA",
        "oleron": "1A2659,455285,7784B7,AAB7E8,D3E0FA,3C5600,7A711F,B79A5E,F1CEA4,FDFDE6",
        "turku": "000000,242420,424235,5F5F44,7E7C52,A99965,CFA67C,EAAD98,FCC7C3,FFE6E6",
    },
    "landcover": {
        "esa": "006400,ffbb22,ffff4c,f096ff,fa0000,b4b4b4,f0f0f0,0064c8,0096a0,00cf75,fae6a0",
        "nlcd": (
            "466b9f,d1def8,dec5c5,ab0000,ab0000,ab0000,b3ac9f,68ab5f,1c5f2c,"
            "b5c58f,af963c,ccb879,dfdfc2,d1d182,a3cc51,82ba9e,dcd939,ab6c28,b8d9eb"
        ),
    },
    "misc": {
        "jet": "00007F,002AFF,00D4FF,7FFF7F,FFD400,FF2A00,7F0000",
    },
}

_GROUP_ORDER = ("colorcet", "cmocean", "crameri", "landcover", "misc")


def _build_palette_presets() -> dict:
    presets = {"Custom": ""}
    for name in sorted(_MATPLOTLIB_PRESETS.keys()):
        presets[name] = _MATPLOTLIB_PRESETS[name]
    for group in _GROUP_ORDER:
        for name in sorted(_PALETTE_BY_GROUP[group].keys()):
            presets[f"{group}:{name}"] = _PALETTE_BY_GROUP[group][name]
    return presets


PALETTE_PRESETS = _build_palette_presets()
