# DukeFootball
statistics created from Duke University football games.

SOME OF THE VARIABLES USED IN THE DATASETS INCLUDE:
OppName      Name of Opposing Team
FPI          Football Power Index during the year listed. Sourced from ESPN.
FPI_diff     Difference between FPI of opposing team and FPI of Duke during the year listed.
Surface	     Playing surface
Day	         Day of the week (Mon, Tue, Wed, etc.)
Start_Time	 Approx. start time of the game. 24-hr format. Minutes represented by decimals. For instance: 12:30 PM is represented by the value 12.5. 9:00 PM is represented by the value 21.0.
Site	       Home, Away, or Neutral
Result	     W for win, L for loss
DukePts	     Points scored by Duke in the game
OppPts	     Points scored by opposing team in the game
PointDiff	   Difference in points scored b/w Duke and opposing team in the game
AttNum	     Number of in-person attendees at the game
AttPct	     Percent of total stadium capacity filled, represented by a decimal between 0 (for 0% full stadium) and 1 (for 100% full).
ESPN_WinPred  Percent chance (at kickoff) that Duke would win the game, per ESPN.
Rain	       If significant rainfall occurred in the game's location/city during the DAY of the game.
1stSeedQB	   TRUE if the first-seed quarterback started the game, FALSE if otherwise
SchoolBreak	 Did the game occur during a break in Duke classes? (TRUE/FALSE)
NatlHoliday	 Did the game occur on the day of a national holiday? (TRUE/FALSE)
TV_Coverage	 the network which aired the game live
City	       City in which the game took place
State	       State in which the game took place (abbreviated)
Bowl	       TRUE if bowl game; FALSE otherwise
UNC_Game     TRUE if played against UNC; FALSE otherwise
