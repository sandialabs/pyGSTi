


class GatestringStructure(object):
    def __init__(self, maxLs, germs, prepfids, effectfids, xy_gatestring_dict, fidpair_filters, gatestring_filters, gatelabel_aliases):
        self.maxLs = maxLs
        self.germs = germs
        self.prepfids = prepfids
        self.effectfids = effectfids
        self.LgermDict = xy_gatestring_dict
        self.fidpair_filters = fidpair_filters
        self.gatestring_filters = gatestring_filters
        self.aliases = gatelabel_aliases

        self.used_maxLs = [ x for x in maxLs if any([ (self.LgermDict[(x,y)] is not None) for y in germs]) ]
        self.used_germs = [ y for y in germs if any([ (self.LgermDict[(x,y)] is not None) for x in maxLs]) ]


        baseStrs = [] # (L,germ) base strings without duplicates
        for L in maxLs:
            for germ in germs:
                if xy_gatestring_dict[(L,germ)] is not None and \
                   xy_gatestring_dict[(L,germ)] not in baseStrs:
                        baseStrs.append( xy_gatestring_dict[(L,germ)] )
        self.baseStrs = baseStrs
