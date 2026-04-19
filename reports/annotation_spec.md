# Annotation Specification: news_topic_classification

## Goal
Assign each news text to exactly one topic class.

## Classes
- **world** — politics, diplomacy, conflicts, international affairs, government actions.
- **sports** — games, athletes, tournaments, leagues, scores, transfers.
- **business** — companies, markets, finance, economic policy, earnings, trade.
- **sci_tech** — technology products, AI, science discoveries, software, hardware, startups.

## Borderline cases
- If article is about a sports club's finances, prefer **sports** if the sporting event/team is central.
- If article is about a tech company's earnings, prefer **business** if the focus is markets/revenue.
- If article is about science policy by a government, prefer **world** if politics dominates.
- If article mentions technology casually but discusses markets, prefer **business**.

## Examples
### world
1. Venezuelans Vote Early in Referendum on Chavez Rule (Reuters) Reuters - Venezuelans turned out early\and in large numbers on Sunday to vote in a historic referendum\that will either remove left-wing President Hugo Chavez from\office or give him a new mandate to govern for the next two\years.
2. S.Koreans Clash with Police on Iraq Troop Dispatch (Reuters) Reuters - South Korean police used water cannon in\central Seoul Sunday to disperse at least 7,000 protesters\urging the government to reverse a controversial decision to\send more troops to Iraq.
3. Palestinians in Israeli Jails Start Hunger Strike (Reuters) Reuters - Thousands of Palestinian\prisoners in Israeli jails began a hunger strike for better\conditions Sunday, but Israel's security minister said he\didn't care if they starved to death.

### sports
1. Phelps, Thorpe Advance in 200 Freestyle (AP) AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.
2. Reds Knock Padres Out of Wild-Card Lead (AP) AP - Wily Mo Pena homered twice and drove in four runs, helping the Cincinnati Reds beat the San Diego Padres 11-5 on Saturday night. San Diego was knocked out of a share of the NL wild-card lead with the loss and Chicago's victory over Los Angeles earlier in the day.
3. Dreaming done, NBA stars awaken to harsh Olympic reality (AFP) AFP - National Basketball Association players trying to win a fourth consecutive Olympic gold medal for the United States have gotten the wake-up call that the "Dream Team" days are done even if supporters have not.

### business
1. Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.
2. Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on another part of the market.
3. Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters - Soaring crude prices plus worries\about the economy and the outlook for earnings are expected to\hang over the stock market next week during the depth of the\summer doldrums.

### sci_tech
1. 'Madden,' 'ESPN' Football Score in Different Ways (Reuters) Reuters - Was absenteeism a little high\on Tuesday among the guys at the office? EA Sports would like\to think it was because "Madden NFL 2005" came out that day,\and some fans of the football simulation are rabid enough to\take a sick day to play it.
2. Group to Propose New High-Speed Wireless Format (Reuters) Reuters - A group of technology companies\including Texas Instruments Inc. (TXN.N), STMicroelectronics\(STM.PA) and Broadcom Corp. (BRCM.O), on Thursday said they\will propose a new wireless networking standard up to 10 times\the speed of the current generation.
3. AOL to Sell Cheap PCs to Minorities and Seniors (Reuters) Reuters - America Online on Thursday said it\plans to sell a low-priced PC targeting low-income and minority\households who agree to sign up for a year of dialup Internet\service.
