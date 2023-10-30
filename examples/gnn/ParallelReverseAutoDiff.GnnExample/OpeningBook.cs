//------------------------------------------------------------------------------
// <copyright file="OpeningBook.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// An opening book of chess openings.
    /// </summary>
    public static class OpeningBook
    {
        /// <summary>
        /// Gets chess openings.
        /// </summary>
        /// <returns>The map.</returns>
        public static Dictionary<string, Move[]> GetOpenings()
        {
            Dictionary<string, Move[]> openings = new Dictionary<string, Move[]>();
            var vienna1 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - b1 - c3}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Vienna1", vienna1);
            var vienna2 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - b1 - c3}"),
                new Move("{bb - f8 - c5}"),
            };
            openings.Add("Vienna2", vienna2);
            var vienna3 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - b1 - c3}"),
                new Move("{bn - b8 - c6}"),
            };
            openings.Add("Vienna3", vienna3);
            var adelaide = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{f2 - f4}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wn - g1 - f3}"),
                new Move("{f7 - f5}"),
            };
            openings.Add("Adelaide", adelaide);
            var belgrade = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Belgrade", belgrade);
            var bishop1 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{f2 - f4}"),
                new Move("{e5 - f4}"),
                new Move("{wb - f1 - c4}"),
            };
            openings.Add("Bishop1", bishop1);
            var bishop2 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Bishop2", bishop2);
            var bishop3 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bn - b8 - c6}"),
            };
            openings.Add("Bishop3", bishop3);
            var bishop4 = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bb - f8 - c5}"),
            };
            openings.Add("Bishop4", bishop4);
            var evans = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bb - f8 - c5}"),
                new Move("{b2 - b4}"),
            };
            openings.Add("Evans", evans);
            var piano = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bb - f8 - c5}"),
            };
            openings.Add("Piano", piano);
            var italian = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - c4}"),
            };
            openings.Add("Italian", italian);
            var italianPolerio = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - f3 - g5}"),
                new Move("{d7 - d5}"),
                new Move("{e4 - d5}"),
                new Move("{bn - c6 - a5}"),
            };
            openings.Add("Italian Polerio", italianPolerio);
            var locock = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d6}"),
                new Move("{d2 - d4}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Locock", locock);
            var lolli = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - f3 - g5}"),
                new Move("{d7 - d5}"),
            };
            openings.Add("Lolli", lolli);
            var lucchini = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - c4}"),
                new Move("{bb - f8 - c5}"),
                new Move("{d2 - d3}"),
                new Move("{f7 - f5}"),
            };
            openings.Add("Lucchini", lucchini);
            var grunfeld = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{c2 - c4}"),
                new Move("{g7 - g6}"),
            };
            openings.Add("Grunfeld", grunfeld);
            var alekhine = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{e4 - e5}"),
                new Move("{bn - f6 - d5}"),
            };
            openings.Add("Alekhine", alekhine);
            var caroKann = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c6}"),
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
            };
            openings.Add("CaroKann", caroKann);
            var sicilian = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d6}"),
            };
            openings.Add("Sicilian", sicilian);
            var sicilianAlapin = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{c2 - c3}"),
                new Move("{d7 - d5}"),
            };
            openings.Add("Sicilian Alapin", sicilianAlapin);
            var sicilianFrench = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{e7 - e6}"),
            };
            openings.Add("Sicilian French", sicilianFrench);
            var ruyLopez = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - b5}"),
            };
            openings.Add("Ruy Lopez", ruyLopez);
            var ruyLopezBerlinRioGambit = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wb - f1 - b5}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wk - e1 - g1 - o-o}"),
                new Move("{bn - f6 - e4}"),
            };
            openings.Add("Ruy Lopez Berlin Rio Gambit", ruyLopezBerlinRioGambit);
            var kingsIndian = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{c2 - c4}"),
                new Move("{g7 - g6}"),
            };
            openings.Add("Kings Indian", kingsIndian);
            var slav = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
                new Move("{c2 - c4}"),
                new Move("{c7 - c6}"),
            };
            openings.Add("Slav", slav);
            var englishOpening = new[]
            {
                new Move("{c2 - c4}"),
                new Move("{e7 - e5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - b8 - c6}"),
            };
            openings.Add("English", englishOpening);
            var nimzoIndian = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{c2 - c4}"),
                new Move("{e7 - e6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{bb - f8 - b4}"),
            };
            openings.Add("Nimzo Indian", nimzoIndian);
            var catalanOpening = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
                new Move("{c2 - c4}"),
                new Move("{e7 - e6}"),
                new Move("{wn - g1 - f3}"),
                new Move("{b7 - b6}"),
            };
            openings.Add("Catalan", catalanOpening);
            var pircDefense = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{d7 - d6}"),
                new Move("{d2 - d4}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Pirc", pircDefense);
            var pircByrne = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{d7 - d6}"),
                new Move("{d2 - d4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{g7 - g6}"),
                new Move("{wb - c1 - g5}"),
            };
            openings.Add("Pirc - Byrne", pircByrne);
            var scandinavianDefense = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{d7 - d5}"),
                new Move("{e4 - d5}"),
                new Move("{bq - d8 - d5}"),
            };
            openings.Add("Scandinavian", scandinavianDefense);
            var frenchDefense = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{e7 - e6}"),
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
            };
            openings.Add("French", frenchDefense);
            var dutchDefense = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{f7 - f5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Dutch", dutchDefense);
            var queensGambitAlbin = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
                new Move("{c2 - c4}"),
                new Move("{e7 - e5}"),
            };
            openings.Add("Queen's Gambit - Albin", queensGambitAlbin);
            var queensGambit = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
                new Move("{c2 - c4}"),
                new Move("{d5 - c4}"),
            };
            openings.Add("Queen's Gambit", queensGambit);
            var queensGambitDeclinedChigorin = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
                new Move("{c2 - c4}"),
                new Move("{bn - b8 - c6}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bb - c8 - g4}"),
            };
            openings.Add("Queen's Gambit Declined - Chigorin", queensGambitDeclinedChigorin);
            var retiOpening = new[]
            {
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Reti", retiOpening);
            var neoGrunfeldUltraDelayedOpening = new[]
            {
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wb - f1 - g2}"),
                new Move("{g7 - g6}"),
                new Move("{d2 - d4}"),
                new Move("{bb - f8 - g7}"),
                new Move("{wk - e1 - g1 - o-o}"),
                new Move("{c7 - c6}"),
                new Move("{c2 - c4}"),
                new Move("{bk - e8 - g8 - o-o}"),
                new Move("{c4 - d5}"),
                new Move("{c6 - d5}"),
            };
            openings.Add("Neo-Grunfeld Ultra-Delayed", neoGrunfeldUltraDelayedOpening);
            var neoGrunfeldOpening = new[]
            {
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wb - f1 - g2}"),
                new Move("{g7 - g6}"),
                new Move("{d2 - d4}"),
                new Move("{bb - f8 - g7}"),
                new Move("{wk - e1 - g1 - o-o}"),
                new Move("{c7 - c6}"),
                new Move("{c2 - c4}"),
                new Move("{bk - e8 - g8 - o-o}"),
            };
            openings.Add("Neo-Grunfeld", neoGrunfeldOpening);
            var queensPawnZukertortOpening = new[]
            {
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wb - f1 - g2}"),
                new Move("{g7 - g6}"),
                new Move("{d2 - d4}"),
                new Move("{bb - f8 - g7}"),
                new Move("{wk - e1 - g1 - o-o}"),
                new Move("{c7 - c6}"),
            };
            openings.Add("Queen's Pawn Zukertort", queensPawnZukertortOpening);
            var queensPawnPseudoCatalanOpening = new[]
            {
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wb - f1 - g2}"),
                new Move("{g7 - g6}"),
                new Move("{d2 - d4}"),
                new Move("{bb - f8 - g7}"),
            };
            openings.Add("Queen's Pawn Pseudo-Catalan", queensPawnPseudoCatalanOpening);
            var kingsIndianAttack = new[]
            {
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d5}"),
                new Move("{g2 - g3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wb - f1 - g2}"),
            };
            openings.Add("King's Indian Attack", kingsIndianAttack);
            var birdsOpeningFromsGambit = new[]
            {
                new Move("{f2 - f4}"),
                new Move("{e7 - e5}"),
                new Move("{f4 - e5}"),
                new Move("{d7 - d6}"),
            };
            openings.Add("Bird - From's Gambit", birdsOpeningFromsGambit);
            var nimzowitschLarsenAttackClassical = new[]
            {
                new Move("{b2 - b3}"),
                new Move("{e7 - e5}"),
                new Move("{wb - c1 - b2}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Nimzowitsch Larsen Attack", nimzowitschLarsenAttackClassical);
            var polishOpeningMainLine = new[]
            {
                new Move("{b2 - b4}"),
                new Move("{e7 - e5}"),
                new Move("{wb - c1 - b2}"),
                new Move("{bb - f8 - b4}"),
            };
            openings.Add("Polish", polishOpeningMainLine);
            var grobOpeningGrobsGambit = new[]
            {
                new Move("{g2 - g4}"),
                new Move("{d7 - d5}"),
                new Move("{g4 - g5}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Grob", grobOpeningGrobsGambit);
            var semiSlavDefense = new[]
            {
                new Move("{d2 - d4}"),
                new Move("{d7 - d5}"),
                new Move("{c2 - c4}"),
                new Move("{c7 - c6}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{e7 - e6}"),
            };
            openings.Add("Semi Slav", semiSlavDefense);
            var sicilianNajdorf = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d6}"),
                new Move("{d2 - d4}"),
                new Move("{c5 - d4}"),
                new Move("{wn - f3 - d4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{a7 - a6}"),
            };
            openings.Add("Sicilian Najdorf", sicilianNajdorf);
            var sicilianScheveningen = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d6}"),
                new Move("{d2 - d4}"),
                new Move("{c5 - d4}"),
                new Move("{wn - f3 - d4}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{e7 - e6}"),
            };
            openings.Add("Sicilian Scheveningen", sicilianScheveningen);
            var sicilianDragon = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d6}"),
                new Move("{d2 - d4}"),
                new Move("{c5 - d4}"),
                new Move("{wn - f3 - d4}"),
                new Move("{g7 - g6}"),
            };
            openings.Add("Sicilian Dragon", sicilianDragon);
            var sicilianAcceleratedDragon = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{g7 - g6}"),
                new Move("{d2 - d4}"),
                new Move("{c5 - d4}"),
                new Move("{wn - f3 - d4}"),
                new Move("{bn - g8 - f6}"),
            };
            openings.Add("Sicilian Accelerated Dragon", sicilianAcceleratedDragon);
            var sicilianSveshnikov = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{bn - g8 - f6}"),
                new Move("{wn - b1 - c3}"),
                new Move("{bn - b8 - c6}"),
                new Move("{d2 - d4}"),
                new Move("{c5 - d4}"),
                new Move("{wn - f3 - d4}"),
                new Move("{e7 - e5}"),
            };
            openings.Add("Sicilian Sveshnikov", sicilianSveshnikov);
            var sicilianPaulsen = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{e7 - e6}"),
                new Move("{d2 - d4}"),
                new Move("{c5 - d4}"),
                new Move("{wn - f3 - d4}"),
                new Move("{a7 - a6}"),
            };
            openings.Add("Sicilian Paulsen", sicilianPaulsen);
            var sicilianCanal = new[]
            {
                new Move("{e2 - e4}"),
                new Move("{c7 - c5}"),
                new Move("{wn - g1 - f3}"),
                new Move("{d7 - d6}"),
                new Move("{wb - f1 - b5}"),
                new Move("{bn - b8 - d7}"),
            };
            openings.Add("Sicilian Canal", sicilianCanal);
            return openings;
        }
    }
}
