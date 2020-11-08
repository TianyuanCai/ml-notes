---
title: Problems
weight: 1
---
# Problems

<!-- Just need a katex comment to generate katex -->

{{< katex >}} {{< /katex >}}

## Product Question Frameworks

Common types of questions

- How to improve
  - Activation
  - User satisfaction
  - Engagement
  - Churn rate
- Test effect of a feature
  - A/B test

Framing the questions/goal

Question: What single feature has the most positive trial-to-sale conversion
impact when looking at users who signed up through social media? Feature goal:
[This feature launch] will result in an uplift of at least 10% when looking at
the conversion rate of trial customers to paying customers over the next 3
months.

Should we implement X? How to improve Z?

Product Analytics Question Framework (Explore) = Feature Launch Framework
(Hypothesis-based)

- Segmented: The question already takes into account a subset of data.
- Detailed: The question is as detailed and specific as possible.
- Actionable: The question can only be answered in a way so that the output can
  be used to take action.

Choose a metric:

- Start from the high-level company goal, then mathematically translate the
  business goal into a metric. In most cases, that is (or should be) growth.
  Then narrow it down:
  - You can grow by increasing user acquisition and user retention
  - I’ll start by looking at which one is below industry average and focus on
    tackling that.
    - You can increase user acquisition by increasing marketing effort and
      identifying pain points in the acquisition funnel
    - You can increase user retention by increasing user engagement
    - Is there anyone that you’d like me to focus on?
  - Define a metric to measure improvement
    - Define:
      - Most metrics need a time threshold - 7 friends in 10 days.
    - Why:
      - I pick metric X because it is related to user engagement and, if I move
        it, I can realistically expect to improve company growth

How to measure and test each metric

Issues with each metric \& de-bias

What distribution do you expect the result data to follow?

What’s the unit of analytics. Is it the same as the unit of diversion?

- For instance: General Unit of Analytics for Company X users
- Ranking: pageview, only sophisticated users will notice
- Look/interface: users, want them to have a consistent experience

- Explain step-by-step how you’ll find the answer if you had the data in front
  of you. They are NOT testing if you are a product visionary.
- When asked to optimize a long term metric like retention rate or lifetime
  value, the question means: find a short term metric that can predict the long
  term one, and then focus on optimizing that one
- When asked to pick variables, pick a combination of user characteristics (age,
  sex, country, etc.) and behavioral ones (related to their browsing behavior)

Pitfalls - Misaligned/incorrect tracking setup - Insufficient knowledge of
product analytics - Treatment of users = sessions - Poor taxonomy/definitions of
terms - Too strong quantitative focus - Non-involvement of key stakeholders

### Business Framework

Core Metrics are a set of metrics that cover the most vital aspects of the
business’ health.

- Core metrics should cover all the most important aspects of your business.
- By monitoring these metrics, and all changes to them, it enables the business
  to constantly be aware of risks and challenges. Core metrics are constantly
  monitored, in the best case with an alerting mechanism.
- They should consist of a combination of leading \& lagging metrics.
  - Leading: Indicate potential change before the impact has a chance to occur
    - Customer Satisfaction
    - Sessions on Website
    - Inbound Volume of Calls
    - Trial Signups
  - Lagging: Shows the impact of the change after the impact has occurred
    - Revenue
    - Net New Sales
    - Churn
    - Monthly Active Users (MAU)

You use 3 core metrics to assess the health of your business, monitored 24/7:

- Customer Acquisition Cost (CAC) = (dollar) Total sales and marketing expenses
  / (\#) new customers acquired = (dollar) CAC
- Customer Lifetime Value (LTV)
  - Calculation
    - For SaaS companies - does not include gross margin
      - (dollar) Average monthly revenue per customer X (\# months) customer
        lifetime = (dollar) LTV
    - For a more precise calculation for LTV, use one of these formulas that
      factor in gross margin:
      - (dollar) Average Order Value X (\#) Repeat Sales X (\# months) Average
        Retention Time X (%) gross margin = (dollar) LTV
  - There are different ways to calculate it. Define well, and calculate.
  - Invest in good customers and focus on boosting their LTV.
  - Customer satisfaction boost LTV.
- $LTV/CAC$: 3 is often considered a minimum
- Churn Rate = \# of Churned Customers/# of Customers who could have churned

  - Required a more detailed definition of each component.
  - Churn is a lagging indicator. It often can be predicted by leading
    indicators such as
    - Usage: Most of the time when the customer is about to churn he’ll stop
      using the product before he cancels. So, if you’re frequently looking at
      the customers’ usage levels, you can act right after a drop happens.
      That’s when you contact this customer to prevent cancelation. In other
      words, you try to get him back on track.
    - Sticky Features: You can track the usage of every different feature of the
      product. We noticed a huge difference in the email marketing usage between
      customers that were staying and customers that were leaving. The
      difference was so huge that it became obvious to us that the customers who
      were not sending email campaigns were about to leave. Therefore, email
      marketing was a sticky feature and we had to encourage them to use it to
      prevent churn.
    - Customer’s KPIs: Customers are leaving because the feature does not meet
      their KPIs comparing to other customers. They are not getting the intended
      benefit of the product.
    - Support Ticket: 70 percent of the churns had never talked to anyone in the
      support team, not a single word. The leading indicator there was silence.
      Exactly, silence is the most anguishing leading indicator. If they are not
      willing to talk to you, they are probably not using the service, or they
      simply don’t care.
  - Calculating the number of churns during the period should NOT include any
    new subscriptions or cancellations for those new subscriptions. Here is an
    example:

    Customers at Start 1800 Existing who leave by end of period 102 New
    Customers 127 New who churn 18 Total churns 120

    The correct way to calculate: 102 / 1800 = 5.67%

    Annual Churn (VERY BAD!): (1 - churn rate) ^ 12 = Annual retention rate (1 -
    .0567) ^ 12 = 0.496361414

Feature Launch: Core Metrics will give you a basis on which to measure these
changes to your product!

Business Plan: 1.5x increase in CAC: CLTV in max. 12 months

-> Feature Launch XYZ: decrease churn by 5% with 60% user adoption rate of the
feature within 3 months of launch

- Finding the right metrics for your business is key
- Every metric, whether leading or lagging, will show you different aspects of
  your business.
- Select a set of core metrics that will give you the best overview of all
  aspects of your business quickly, easily, and constantly.

#### User \& Cohort Analysis

- Metrics can be created based on user, session or event scopes, or a
  combination of them
- User Cohorts (segmentation) is necessary and a powerful way to understand your
  users
- Cohort analysis is the most effective method of analyzing retention for any
  metric
- User analysis

- User Segmentation
  - Example selection basis
    - Usage: clicked “recommend to a friend” in your product
    - Attribution: purchased your product in the last 7 days
    - Geographic: users from Antarctica
    - Demographic: users whose company size is over 50 employees
    - Psychographic: sentiment analysis where the customer is “positive”
    - Behavioural: analysis shows them as “price-conscious”
  - Process
    - First we select the basis of the cohort. This forms the initial grouping
      for further analysis.
      - E.g. the user’s signup date to our product
    - Now we select the time frame of the cohort. This is typically smaller for
      more detailed analysis, so let’s start with a large grouping.
      - E.g. monthly
    - Now we select how we want to analyze the cohorts. This is the most
      interesting part, as we decide what information we are trying to
      understand through this analysis.
      - E.g. recommended our product to a friend
    - Now we select the date range for the entire analysis. Without this, we
      cannot frame the analysis so we need to limit the data.
      - E.g. last 12 months
    - Final
      - All users grouped by
      - signup date to our product
      - who recommended our product to a friend
      - grouped by month
      - for the last 12 months

#### Metric Sets

Very rarely will one metric hold the key to understanding how your business is
doing.

A good metric set

- Focus on 3 - 5
- cover the entire spectrum of the business
- integrate or complement your feature goals
- are flexible and can change for the needs of the business
- You can always go deeper when looking into data; keep your metric set suitably
  high-level to yield insights across the business
- Ensuring actions can be taken from the metric set is more important than
  having all the metrics in one set

Case example

- online car insurance configuration \& calculation tool that is sold in a
  subscription model for other businesses to use (B2B2C)
- customers can test the product for free for 14 days
- 3 subscription tiers (basic, pro, ultimate)

Metric Set

- CAC: CLTV
- Churn Rate
- \# of Monthly Active Users
- \# of New Subscriptions
- Trial: Sale Conversion Rate
- Form Completion Rate
- Form Errors per Non-Complete

#### Visual Story-telling

- K.I.S.S. (keep it simple, stupid)

  the more complicated the report, the more you will have to explain it to
  people

- Know your audience

  Each report/dashboard should be tailored to a certain audience

- Visual storytelling

  Use best practices when visualizing data

  Common mistakes

- Not using color for communication. Not using enough dimensions when
  representing data.
- Use a stacked line chart to compare volume rather than a bar chart. The bar
  chart with two groups side by side will be way easier.
- Different scales
- Unnecessarily distort Y-axis.
- Merging data of different scopes/levels

Chart Suggestion - a thought-starter

#### Monitoring

- Automate reporting and alarming
- variable or fixed thresholds; each depends on your metric and business need,
  but without them, you cannot alert
- defining how to notify the relevant stakeholders is important to be able to
  take action

Setup

- Systems are less important than reliability and automation
- Ensure that thresholds can be set automatically but can also be adjusted
- Make all stakeholders in the value chain aware of the setup:
  - DevOps: will most likely also receive alerts
  - Developers: may see the alerts as well
  - You and your teammates: (obviously)
  - Your ‘boss’: explain the value
  - Management: if necessary, depending on company size, operational impact, and
    maturity of data culture

#### Data Blending

Be careful merging data of different scopes!

- Which users are taxing your support team the most, in proportion to the length
  of time as a customer?
  - e.g. Ticketing System + CRM
- What discount percentage is given across which acquisition channels?
  - e.g. Subscription Management + Web Analytics
- Which email campaigns/flows have the best effect on lowering subscription
  cancellation rates?
  - e.g. Emailing + Product Analytics + Subscription Management
- “How has the adoption of this feature varied across our customer journey from
  trial to paying user against the goals we set?”
  - Web Analytics
    - Campaigns: to determine which channels worked best when advertising this
      new feature
  - Product Analytics
    - Usage: to look at the feature adoption and the evaluation of our feature
      launch goals
  - Subscription Management
    - Financials: to show which customers converted from trial to paying (and
      their financials)
  - (optionally) Ticketing System
    - Customer feedback: show satisfaction rates and ticket counts for questions
      about this feature

### Product Metrics

#### General AARRM Framework

- Acquisition
- Activation
- Engagement / Retention rate
- Referral (especially to external sites)
- Monetization / Conversion

#### UX Specific Metrics

- Product
  - Social network strength
  - Similar products / duplicates matching
  - Monetization
  - User success
  - Rank
- Feed/Search UX: To help users easily identify the most relevant and quality
  content. Entry Relevance \& Quality
  - Relevance: Click-to-expand probability [F]
  - Relevance \& Lazy Engagement: Click-through probability [F,S]
  - Relevance \& Quality [F, S, User Growth]

#### Acquisition

- Ranking accuracy
  - Average First-click rank (>= 1, the lower the better)
    - Distribution is highly skewed
    - Use permutation test to get CI and test hypothesis
    - Use a non-parametric method to generate CI and test hypotheses
    - In case the user does not click, use the length of list + 1 as the metric.
      We can consider log transforming the data to make it mimic the decreasing
      value to the user. Average time spent on the feed (need to be winsorized)
- Engagement - DAU, WAU, MAU
- Ads (All products)
  - Click-through probability _At least one click / Pageview, Unit - Pageview_
  - CPM/CPC
  - Monthly Recurring Revenue
  - Average revenue per user
- User Acquisitions
  - Monthly Active User (MAU): Users who have logged in to your product within
    the last 30 days
  - Trial Users: Users who have signed up for your product’s free trial
  - Paying Users: Users who have paid for your product
  - Feature Adoption Rate: % of users who are using the feature (definition of
    ‘usage’ is flexible)
- User growth
  - Hard to measure or negligible ones:
    - Customer acquisition cost, LTV, Payback period
  * Activation rate

#### Engagement

“Engaged” - users who have clicked-through and performed an action

- Apply the change to a new group of users who’ve just completed lazy signup.
  Intuition - Expose them to the changes and see whether they convert within a
  given period.

- Time spent / user attention
  - Session length
  - Duration of impressions
  - Number of impressions that is longer than 3 seconds
  - Impressions: 20-second impression for reading a post Metrics by product
    areas
- Any pages
  - Click-through rate (on a given content)
  - Bounce Rate
- Content Quality
  - Click-through rate
  - Answers: Click-through \& Upvote/Bookmark/Share rate
  - Questions: Click-through \& Follow/Share/Request rate
  - Upvotes/downvotes (only downvotes for questions)
  - Bounce Rate
  - Number of edits
  - View count
- Item Quality
  - \# Merged questions
  - Ask frequency
  - Answer requests
  - Follower number
  - Unique visitors / read count
  - Number of answers
  - Number of sent notifications for related users
  - Number of topic to which a question belongs
- General
  - Number of Inbound and Outbound links
- Users
  - Number of pages viewed
  - Session duration

### Common Biases

#### Clicks

Click-through rate has a bias - doesn’t necessarily measure what we want. Maybe
novelty, catchy names, but doesn’t always measure quality content. Hard to
measure clicks because of connection speed, latency, and other issues.

Click-through rate/probability

- CTR usually measure the visibility(button in our example)
  - Total number of clicks/person = 5/2 = 2.5
- CTP measures the impact. CTP will avoid the problem when users click more than
  once. CTP can be acquired when working with engineers, to capture the number
  of clicks every change, and get CTR with at least one click.
  - At least one/person = 0.5

#### Ranks

- Position bias
  - When clicking - users tend to favor the top items although these might be
    less relevant to them. Therefore, when looking at rank data What if the user
    did not click. How to determine the position when that happens? Solution - A
    prerequisite of training ranking algorithm using click data is to debias it.
    - When training - Give top answers less weight so that it provides less
      information than clicks on entries that are farther down on the list.
      Existing approaches use search result randomization over a small
      percentage of production traffic to estimate the position bias. This is
      not desirable because result randomization can negatively impact users’
      search experience.
    - Use interaction term

#### Primacy \& Novelty Effects

- Novelty effect. Might’ve not captured the long-term impact on the business and
  users.
  - When a feature has suspected novelty effects you can compare only its effect
    on the new users in both groups.
- Primacy effect. Primacy effects are those where initial treatment effects
  appear small in the beginning, but increase over time (e.g., as a machine
  learning system better adapts).
  - When a feature has suspected primacy effects you can do a ‘warm start’ so
    that the marginal improvement in performance over time is smaller.

#### Simpson’s Paradox - segment misinterpretation

- Simpson’s Paradox is a phenomenon that occurs when we observe a certain
  directional trend or relationship in all mutually exclusive segments of the
  data, but the same trend is not observed (or reverses) when we look at the
  combined dataset.

Example here, of the user table

- It happens because cells that have more data dominate the totals for their
  corresponding variants and are capable of switching the direction of overall
  results. Both Browsers row is dominated by Firefox Browser for variant A
  (87.5% success rate) while it is dominated by Chrome Browser for variant B
  (62.5% success rate).
- You can prevent this problem from happening in two ways:
- Stratified sampling, which is the process of dividing members of the
  population into homogeneous and mutually exclusive subgroups before sampling.
  The condition used for defining a segment must not be impacted by the
  treatment.

#### Other common biases

- Biases
  - Assuming underpowered metrics had no change
  - Claiming success with a borderline p-value, need to test again.
  - Continuous monitoring and early stopping
  - Assuming metric movements are consistent in different subsets of the
    treatment group.
  - Novelty \& Primacy effects
  - Incomplete funnel metric (always include both conditional and unconditional
    success rates)
  - Correlated observations
  - Ignored seasonality

## Cases

### A/B testing

- How to implement A/B testing in a given scenario?
- Why is A/B testing necessary?
  - Why: control for the effects of everything else and only test for the effect
    of the particular change being studied.
  - A/B testing allows individuals to make careful changes to their user
    experiences while collecting data on the results. This allows them to
    construct hypotheses, and to learn better why certain elements of their
    experiences impact user behavior.
  - Testing one change at a time helps them pinpoint which changes had an effect
    on their visitors’ behavior, and which ones did not. The outside world often
    has a much larger effect on metrics than product changes do. Users can
    behave very differently depending on the day of the week, the time of year,
    the weather (especially in the case of a travel company like Airbnb), or
    whether they learned about the website through an online ad or found the
    site organically. Controlled experiments isolate the impact of the product
    change while controlling for the aforementioned external factors.
  - Background: A/B testing is essentially an experiment where two or more
    variants of a page are shown to users at random, and statistical analysis is
    used to determine which variation performs better for a given conversion
    goal.
  - Why: control for the effects of everything else and only test for the effect
    of the particular change being studied. A/B testing allows individuals to
    make careful changes to their user experiences while collecting data on the
    results. This allows them to construct hypotheses, and to learn better why
    certain elements of their experiences impact user behavior. Testing one
    change at a time helps them pinpoint which changes had an effect on their
    visitors’ behavior, and which ones did not. The outside world often has a
    much larger effect on metrics than product changes do. Users can behave very
    differently depending on the day of the week, the time of year, the weather
    (especially in the case of a travel company like Airbnb), or whether they
    learned about the website through an online ad or found the site
    organically. Controlled experiments isolate the impact of the product change
    while controlling for the aforementioned external factors.
- How to know significance after A/B testing?
  - Calculation: p-value. Given the null (control) is true, what is the chance
    of observing something equal to or more extreme than the observed. If the
    chance is very low, then we know that the treatment has a statistically
    significant effect. What to do if we have multiple metrics
  - It’s important to have one key metric but also track other metrics to make
    sure the treatment doesn’t impact other metrics negatively. This lets us see
    not only if the treatment has the hypothesized effects but also the
    tradeoffs associated with introducing this treatment.
  - Calculation: p-value. Given the null (control) is true, what is the chance
    of observing something equal to or more extreme than the observed. If the
    chance is very low, then we know that the treatment has a statistically
    significant effect. When to use a t-test and when z-test?
  - Z-test is used when the sample size is large, i.e. n > 30, and the t-test is
    appropriate when the size of the sample is small, in the sense that n ? 30.
- Suppose an experiment lasts 7 days, would you compute the p-value daily or on
  aggregate, why?

  - But the most important reason is that you are performing a statistical test
    every time you compute a p-value and the more you do it, the more likely you
    are to find an effect. The test won’t have enough power if we calculate the
    p-value daily, and as a result, we are more likely to miss the real effect.
    - Power = 1 - Type II Error (False Negative)
    - 1- Power: What percentage of the time are we willing to miss a real
      effect? Typically 80% of the power.
    - Significance: What percentage of the time are we willing to be fooled into
      seeing an effect by random chance? This is called the significance level,
      and more precisely, we would state this as the probability of rejecting
      the null hypothesis. Typically 5%.
  - Since the statistical test is a function of the sample and effect sizes if
    the early effect size is large through natural variation it is likely for
    the p-value to be below 0.05 early.
  - In an A/B testing situation, if you fail to detect a statistically
    significant effect of a given size, you can’t say that there is no effect if
    your test has low power. You didn’t give your test variant any chance. If
    the power is high though, then you can say that with high certainty there is
    no true effect equal to or larger than the effect size the test was powered
    for (has a particular level of sensitivity towards).

- Problem with using \#clicks/#pageviews but release the feature by users? The
  unit of analytics is pageviews
  - The unit of diversion is users
  - Unit of diversion > unit of analytics. Therefore, the observations from the
    same users are correlated, increasing the variance of the data. This will
    result in the need for a larger sample and a longer duration to power the
    experiment.
  - Cases when the unit of analytics is a cookie (say measure \# of clicks per
    cookie per day) but the unit of diversion is the pageview. This will result
    in a cookie being assigned to different groups of experiments. In this case,
    the metric is not well-defined. This causes cross-bucketing.
- Why is average clicks per user for an instant search result, not a great
  metric?
  - Problem with tracking clicks
    - Novelty effect. Might’ve not captured the long-term effectiveness of the
      feature. Clicking doesn’t mean it’s useful for the users. Click and
      engagement and repeated uses provide more evidence.
    - Primacy effect
- When ranking, what would happen if we only optimize for CTR or upvotes?
  - CTR measures relevancy/interestingness/popularity, and Upvotes measure
    quality as well as popularity. When looking at upvotes, it’s better to look
    at the upvotes-to-views ratio. Link not only has CTR but also upvotes so
    this complicates things. I will assume that users engage with Q\&A most of
    the time and the effect of links is relatively small.
  - Analysis
    - Only optimizing CTR results in too much clickbait. Only optimizing upvote
      results in the lack of attention to some niche topics. Only optimizing the
      upvote-to-views ratio results in users maximize social affirmation by
      seeking consensus rather than discussions, which would have engaged more
      users. Users would look for a combination of all four things (in various
      weights) when coming to Company X, interestingness/affinity to the
      questions, a variety of topics - including the niche ones, agreeability,
      and vibrant discussions.
    - All these metrics are also time sensitive so we need to adjust for that.
- Assumptions of the t-test?
  - Continuous/ordinal dependent variable
  - Each observation of the dependent variable is independent of the other
    observations of the dependent variable (its probability distribution isn't
    affected by their values). That the data is collected from a representative,
    randomly selected portion of the total population.
    - Exception: For the paired t-test, we only require that the
      pair-differences (Ai - Bi) be independent of each other (across i).
      "independent" and "dependent" are used in two different senses here. Just
      think of a "dependent variable" as one thing, and "observations that are
      dependent" as another thing.
  - Dependent variable has a normal distribution, with the same variance,
    $\sigma$ 2, in each group (as though the distribution for group A were
    merely shifted over to become the distribution for group B, without changing
    shape)
- What if the data in the t-test is highly skewed?

Permutation t-test

How to do permutation test

- Assumption of the binomial distribution
  - 2 types of outcome
  - Independent events
    - Drawing cards without replacement from a shuffled deck of 20 is not
      binomial. Identical distribution (p is the same for all observations)
- Calculate p-value on a weekly/daily basis?
  - Don't report p-value till we've reached the end of the experiment where the
    sample size requirement is satisfied because calculating p-value without
    reaching the experiment criteria can result in misleading results. You are
    performing a statistical test every time you compute a p-value and the more
    you do it, the more likely you are to find an effect. Power = 1 - Type II
    Error (False Negative)

Power: What percentage of the time are we willing to miss a real effect?
Typically 80% of the power. Significance: What percentage of the time are we
willing to be fooled into seeing an effect by random chance? This is called the
significance level, and more precisely, we would state this as the probability
of rejecting the null hypothesis. Typically 5%. One way to adapt is to adjust
the p-value to account for the extra checking - Bonferroni correction for
example.

- If you can only collect one set of data a day, and you’ve run your experiment
  for multiple weeks. Does it have any conflict with t-test assumptions?

- Problems with calculating p-value in a week?
  - Autocorrelation in time series data. Problem with t-test on time-series data
  - The standard setting in which a t-test is used would require independence
    across observations. Meanwhile, your observations may have some time
    patterns (autocorrelation, etc), if I understand it correctly. Then you
    would need some adjustment to account for that instead of using the
    plain-vanilla t-test.
- How to report if it’s significant
  - Sanity check. Check if the metric movement is homogeneous. Sometimes
    treatments have different effects on different subsets of the treatment
    groups (e.g., users in different countries, or the feature not working
    correctly in certain browsers). If you’re not aware of such effects you may
    end up shipping a feature that improves user experience overall, but
    substantially hurts the experience for a group of users.
  - If we have more time and capacity, and this is an important change, I would
    run another experiment to see if I can replicate the result. I’ll report the
    significance to the stakeholders to help them with decision making. How to
    report if the result is not significant
  - Also, understand the gain we can get from the minimum detectable effect and
    the cost of implementing the feature. Discuss with the product manager to
    determine if it’s worth it to roll out the feature.
- The treatment does not have a statistically significant effect on user growth.
  (Check the Udacity course)
  - Communicate to the decision-makers when they're going to have to make a
    judgment because the data is uncertain. They're going to have to use other
    factors, like strategic business issues, or other factors besides the data.
  - Use other data sources that can complement the experiment
    - External data
    - UER
    - Focus Group
    - Survey
    - Retrospective Analysis
    - Experiments
  - If it’s not significant when it really should
    - You could subset your experiment by platform, time (day of the week) see
      what went wrong or different significant if subset by those features. It
      could lead you to a new hypothesis test and understand how your
      participants react. If you just begin in your experiment, you should
      cross-check your parametric hypothesis and non-parametric hypothesis test.
- How to report if the result is significant for some of the countries? Or in
  general insignificant? Look at segments to see if we can identify causes -
  beware of problems with multiple testing. Use Bonferroni correction. Problems
  with dividing data by countries and testing significance? (Why multiple
  testing could be a problem)
  - Report to the team, but also make them aware of the limitation of the
    analysis - due to the smaller power when looking at the subgroup, the effect
    is much more likely due to chances. The results can be misleading, there are
    possibilities of false positive or false negative conclusions. Consider a
    case where you have 20 hypotheses to test, and a significance level of 0.05.
    What’s the probability of observing at least one significant result just due
    to chance?
    - The Bonferroni correction (conservative) sets the significance cut-off at
      $\alpha$/n. For example, in the example above, with 20 tests and $\alpha$
      = 0.05, you’d only reject a null hypothesis if the p-value is less than
      0.0025. The Bonferroni correction tends to be a bit too conservative when
      the tests are correlated. When they are independent, it tends to be
      slightly lower than the desired significance level.
- Why time spent on the homepage can be good and bad?
  - Good
    - Homepage has the most curated feed, which helps Company X's users easily
      find interesting content. User satisfaction due to the relevancy and
      interestingness of the content
    - Generate the most value - most stickiness and therefore most valuable ads
    - Provide valuable data on users’ ranking of content to help Company X
      improve its learn-to-rank algorithms
  - Bad
    - Not spending enough time interacting with the community
      - Reviewing feed is passive. Users don’t actively engage with other
        members of the community.
      - It’s when they enter the question page and are exposed to more
        interactions that their active interactions happen. These interactions
        benefit the community. Relevant content and meaningful social
        interactions are equally important to the Company X community. The
        amount of user preferences we can learn from their behavior on the
        homepage is limited because they are only indicating the order of their
        preferences in a list we curated. We can get more information and
        provide better prediction when the user leaves the feed and explore a
        wider range of content. E.g. the feed might be fairly constrained when
        it comes to the topics it covers. Maybe users are also an expert or
        interested in other topics. We won’t know until they leave the feed and
        check out that other topic. Lift
  - The difference in response rate between treatment and control. Uplift
    modeling can be defined as improving (upping) lift through predictive
    modeling.
- How to deal with outliers?
- When there are primacy effect and novelty effect, how to detect them?
  - Primacy, customer segmentation and compare the behavior of the experienced
    users to those of the less experienced.
  - Novelty, look at completely new users. How do they behave?
- T-test vs z-test vs Pearson chi-sq test vs fisher’s exact test
  - We use these tests for different reasons and under different circumstances.
  - z-test. A z-test assumes that our observations are independently drawn from
    a Normal distribution with unknown mean and known variance. A z-test is used
    primarily when we have quantitative data. (i.e. weights of rodents, ages of
    individuals, systolic blood pressure, etc.) However, z-tests can also be
    used when interested in proportions. (i.e. the proportion of people who get
    at least eight hours of sleep, etc.)
  - t-test. A t-test assumes that our observations are independently drawn from
    a Normal distribution with unknown mean and unknown variance. Note that with
    a t-test, we do not know the population variance. This is far more common
    than knowing the population variance, so a t-test is generally more
    appropriate than a z-test, but practically there will be little difference
    between the two if sample sizes are large.
  - With z- and t-tests, your alternative hypothesis will be that your
    population mean (or population proportion) of one group is either not equal,
    less than, or greater than the population mean (or proportion) or the other
    group. This will depend on the type of analysis you seek to do, but your
    null and alternative hypotheses directly compare the means/proportions of
    the two groups.
  - Chi-squared test. Whereas z- and t-tests concern quantitative data (or
    proportions in the case of z), chi-squared tests are appropriate for
    qualitative data. Again, the assumption is that observations are independent
    of one another. In this case, you aren't seeking a particular relationship.
    Your null hypothesis is that no relationship exists between variable one and
    variable two. Your alternative hypothesis is that a relationship does exist.
    This doesn't give you specifics as to how this relationship exists (i.e. In
    which direction does the relationship go) but it will provide evidence that
    a relationship does (or does not) exist between your independent variable
    and your groups.
    - How about the chi-sq test on variance? Fisher's exact test. One drawback
      of the chi-squared test is that it is asymptotic. This means that the
      p-value is accurate for very large sample sizes. However, if your sample
      sizes are small, then the p-value may not be quite accurate. As such,
      Fisher's exact test allows you to exactly calculate the p-value of your
      data and not rely on approximations that will be poor if your sample sizes
      are small.
  - I keep discussing sample sizes - different references will give you
    different metrics as to when your samples are large enough. I would just
    find a reputable source, look at their rule, and apply their rule to find
    the test you want. I would not "shop around," so to speak, until you find a
    rule that you "like."
  - Ultimately, the test you choose should be based on a) your sample size and
    b) what form you want your hypotheses to take. If you are looking for a
    specific effect from your A/B test (for example, my B group has higher test
    scores), then I would opt for a z-test or t-test, pending sample size and
    the knowledge of the population variance. If you want to show that a
    relationship merely exists (for example, my A group and B group are
    different based on the independent variable but I don't care which group has
    higher scores), then the chi-squared or Fisher's exact test is appropriate,
    depending on sample size.

#### Test Design

E.g. How to use CTR to measure which ranking algorithm is better?

- Success metric
  - Define experiment parameters. These include the metrics we want to use to
    measure the experiment, the unit of the experiment (Company X can measure by
    user), minimum detectable effect, the margin of error, sensitivity.
    Invariant parameters - what we want to use as our sanity check
  - Guardrail metrics - what metrics we want to protect
  - Calculate the sample size needed, and therefore, how long the experiment
    should run.
- Subject
  - Select population: U.S., Desktop, …
  - Unit of diversion: Pageview
  - Slice on different dimensions and make sure the groups are comparable on
    these different dimensions. Also, make sure the sample is a good
    representation of the population. (How to ensure which user falls into which
    experiment group?)
- Pre-period: A/A experiment to make sure the variability of the metric is
  suitable for our need.
  - A/A testing to make sure the metric is not too variable. Make sure the
    metric is not introducing too much variability - not too sensitive.
- Choose one metric

  - Relevance \& Lazy Engagement: Click-through probability. At least one unique
    click / Pageview, Unit - Pageview, \# of unique clicks / \# of entries
    shown, Unit - Pageview

  - Average rank of first-click (Always greater than 1, the lower the better)

- Randomization
  - How to randomize subjects? Slice on different dimensions and make sure the
    groups are comparable to these different dimensions. Also, make sure the
    sample is a good representation of the population. (How to ensure which user
    falls into which experiment group?)
    - Is it always better to do stratified sampling based on different user
      attributes?
- Test
  - Which test
  - Define test parameters
    - Calculate sample size
  - Know significance

### Product Modeling

Overall Case Framework

- Reiterate the question What is the goal of this exercise? “Success”? How do
  you define success? What kind of data do we have that can help us solve this
  problem? Based on the data, what kind of model can we use to solve this
  problem? Why is this model suitable Pro and Con Potential biases and variances
  of this model? If so, how to correct it? Assumptions / conditions necessary
  - Areas of potential improvement Computation/implementation
- What data can we leverage to answer this question?
  - Outcome variables
  - Features
  - How to incorporate existing Company X data as features?
- How do you test the model for accuracy? How do you compare models’
  performance?
  - Error metrics
  - Model performance metrics
- How do you overcome overfitting?
  - CV, Bagging, Bootstrapping
- How computationally intensive is this model?
  - Know how and why they are intensive? How to expand this model to other areas
    of the product?
- How to know the algorithm benefited the users?
  - Common Metrics
    - Distribution of these metrics. Normal/skewed/multimodal? How to correct in
      each case?
    - Edge cases - what if users did not perform an action?
  - How would you perform an A/B test on this metric?
    - A/B testing experiment framework
  - What are the guardrail metrics?
    - Think

#### Example (Ranking)

Goal: Feed: Given a potential feed story, how good is it?

- Search: Given a potential search result, how relevant is it?
  - Everything is ranked at the time the user visits the feed. This allows us to
    make social updates and ensure the users have exposed their friends’ latest
    activities and latest trends on Company X. Always aim to show the most
    important content on the top of the feed because people are twice as likely
    to interact with what’s on the top.
- Important factors: speed, live updates, feature elimination
  - Not all available features can be used for ranking, and, much of the time,
    only a small percentage of these features can be used. Thus, it is crucial
    to have a feature selection mechanism that can find a subset of features
    that both meets latency requirements and achieves high relevance. Edge case:
    What if we don’t have a lot of information on a new user? We use their
    behavior from their first session Performance Metrics
- Search: Average rank of the last click where user accessed information
  - This was the last relevant query that the user has clicked. Assuming the
    users’ purpose was to find relevant information, the last click means that
    they were reasonably satisfied with the answer they got. Feed
  - [Normalized Discounted Cumulative Gain](https://www.ebayinc.com/stories/blogs/tech/measuring-search-relevance/)
    - DCG measures the usefulness, or gain, of a document based on its position
      in the result list. The gain is accumulated from the top of the result
      list to the bottom, with the gain of each result discounted at lower
      ranks.
    - Assumptions
      - Highly relevant documents are more useful when appearing earlier in a
        search engine result list (have higher ranks)
      - Highly relevant documents are more useful than marginally relevant
        documents, which are in turn more useful than non-relevant documents.
        Scoring based ranking - Boosted decision trees stacked with logistic
        regression
- [getstream.io/blog/beyond-edgerank-personalized-news-feeds/](https://getstream.io/blog/beyond-edgerank-personalized-news-feeds/)
  Video [link](https://www.youtube.com/watch?v=iXKR3HE-m8c)
  [slides](https://www.slideshare.net/SessionsEvents/ewa-dominowska-engineering-manager-facebook-at-mlconf-sea-52016)
- Boosted decision trees stacked with logistic regression
  - Using boosted decision trees allows us to throw (almost) as many features as
    we want at the model and let it decide what is important. After the decision
    trees are trained we can take each input feature and turn it into a set of
    categorical features. We then use those categorical features as one-hot
    encoded inputs into a logistic regression to relearn the leaf weights. This
    extra step allows us to speed up the training process and push the training
    step to a near-online model, while still taking advantage of the feature
    transforms that the boosted trees provided us.
  - Express as the probability of click, like, comment, etc. Assign different
    weights to different events, according to the significance
- Example: close coworker feels an earthquake
  - Highest chance of a click
  - Decent chance of like/comment
- This is good because
  - Use machine learning to measure something directly measurable (objective
    function can be clearly defined. Models are trained on their data. Clicks on
    click data, and like on like data, comment on comment data. How do we
    combine these models?
  - Different teams working on different models. Allow different adjustments to
    be independent of each other. Allow integration of heterogeneous items now
    that we also have shared links and activities from spaces
- Model
  - Boosting Decision trees
    - Start with 100k> potential features. Prune the data to the top 2k. Remove
      the least important feature. Historical counts and propensity are some of
      the strongest features. We then apply time decay. This takes a long time.
      These wouldn’t work if we want to be quick and make decisions based on
      real-time data.
  - Logistic regressions
    - React quickly and incorporate new content
    - Simple, fast, easy to distribute
    - Treat the trees as feature transforms, each one turning the input features
      into a set of categorical features, one per tree. Use logistic regression
      for online learning to quickly re-learn leaf weights. Use the rich outcome
      from boosted trees, convert them to categorical features, and input them
      into decision trees. We can do stacking. Learnings
    - Data freshness matters - simple models allow us to tweak feed live online.
      Feature generation is part of the modeling process
    - Stacking
      - Support plugging in new algorithm and features easily
      - Works well in practice
    - Use skewed sampling to manage high data volume. Historical counters as
      features provide highly predictive features. Easy to update online. How to
      upsample/undersample?

Objective function

- How to define the right objective function? Implicit: Engagement -> Explicit:
  click
  - We can understand implicit metrics through surveying
  - Pair-wise comparison survey. Ask them two stories to see which one is better
  - In-out survey, do you want this in your feed or not?
  - Rating survey - do you want to see this story in your newsfeed? (5 options)
  - Problem is this does not provide context but people make decisions in
    context. Therefore, do this survey in context. Last but not least - weighing
    events
- Weighing events, such as likes and comments for each person. Calibrate these
  weights for each person. Miscellaneous Experiment design to test a new
  application of home page, what metrics would you look at?

What metrics to track for the performance of Company X’s (new) homepage?

- User needs: This is an information sieve problem. The users want to
  efficiently access the quality feeds that they are most interested in.
  Therefore, the key metrics can be measured by users’ engagement on the page
- Contents are interesting in the first sight
  - Click-through rate / Click-to-expand rate
  - Number of minutes spent on the homepage
- Proportion of interesting content
  - Answers: Click-through \& Upvote (or downvote)/Bookmark/Share rate
  - Questions: Click-through \& Follow/Share/Request rate as well as a set of
    metrics for Answers
- The metrics that might lead to better user responses on the homepage
- Questions and Answers: quality, relevancy, novelty, uniqueness.
- Page organization: prioritization
- Similar products/duplicates matching
- If we want to make the search bar longer, what metrics should we look at to
  see its performance?
  - Intuition
    - The longer the length of the search
    - The more detailed the subject of the search will be
    - The more likely the user will find a similar question or ask a unique
      question
  - Proportion of times user find what they are looking for
  - Proportion of duplicated questions asked (decrease)
  - Proportion of unique questions asked (increase)
- Why time spent on the homepage can be good and bad?
  - Good
    - Homepage has the most curated feed, which helps Company X users easily
      find interesting content. User satisfaction due to the relevancy and
      interestingness of the content
    - Generate the most value - most stickiness and therefore most valuable ads
    - Provide valuable data on users’ ranking of content to help Company X
      improve its learn-to-rank algorithms
  - Bad
    - Not spending enough time interacting with the community
      - Reviewing feed is passive. Users don’t actively engage with other
        members of the community.
      - It’s when they enter the question page and are exposed to more
        interactions that their active interactions happen. These interactions
        benefit the community. Relevant content and meaningful social
        interactions are equally important to the Company X community. The
        amount of user preferences we can learn from their behavior on the
        homepage is limited because they are only indicating the order of their
        preferences in a list we curated. We can get more information and
        provide better prediction when the user leaves feed and explore a wider
        range of content. E.g. the feed might be fairly constrained when it
        comes to the topics it covers. Maybe users are also an expert or
        interested in other topics. We won’t know until they leave the feed and
        check out that other topic. Lift
  - The difference in response rate between treatment and control. Uplift
    modeling can be defined as improving (upping) lift through predictive
    modeling.
- How to calculate the similarity between two given questions?

Design a model that detects whether the original Company X question is modified
to a different one?
[TF-IDF](https://www.wikiwand.com/en/Tf%E2%80%93idf#/Definition): word
importance

- Edit distance: minimum numbers of operations required to transform one string
  into the other. Autocorrect selects words from a dictionary that have a low
  distance to the word in question.
- [LDA](https://www.wikiwand.com/en/Latent_Dirichlet_allocation#/Model): topic
