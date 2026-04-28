# **Flight Pricing Engine**

README & Developer Guide

Kamran Sohrab


A dynamic pricing system for general aviation using logistic regression, constrained optimization, and

Q-learning.


**Current** **Versions**


**Project** **V1.3** Updated April 21, 2026

**README** **V2.2** Updated April 6, 2026


1


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Contents**


**1** **Project** **Overview** **4**


**2** **Features** **4**


**3** **System** **Requirements** **5**
3.1 Python Version . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2 Required Libraries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5


**4** **Getting** **Started:** **Setup** **Guide** **5**
4.1 Step 1   - Verify Python is Installed . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
4.2 Step 2   - Download the Project Files . . . . . . . . . . . . . . . . . . . . . . . . . . 5
4.3 Step 3   - Create a Virtual Environment (Recommended) . . . . . . . . . . . . . . . 6
4.4 Step 4   - Install Dependencies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4.5 Step 5   - Configure Your Aircraft Parameters . . . . . . . . . . . . . . . . . . . . . . 6
4.6 Step 6   - Run the Application . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
4.7 Setting Up in an IDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
4.7.1 Visual Studio Code . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
4.7.2 PyCharm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8


**5** **Using** **the** **Application** **8**
5.1 Pricing a Flight   - Input Reference . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
5.2 Reading the Output . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
5.3 Recording the Outcome . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
5.4 Viewing the Analytics Report . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9


**6** **Project** **Structure** **9**
6.1 Data Flow Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10


**7** **Configuration** **Reference** **10**
7.1 Aircraft Limits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
7.2 Pilot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.3 Operating Costs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.4 Pricing Bounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.5 Logistic Regression Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.6 MDP / Q-Learning Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . 12


**8** **Developer** **Guide** **12**
8.1 Architecture Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
8.2 Module-by-Module Reference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
8.2.1 `config.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
8.2.2 `demand` `model.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
8.2.3 `optimizer.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
8.2.4 `mdp` ~~`a`~~ `gent.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
8.2.5 `history.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
8.2.6 `pricer.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
8.2.7 `utils.py` . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
8.3 Adding a New Feature to the Model . . . . . . . . . . . . . . . . . . . . . . . . . . . 15


2


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


8.4 Resetting the Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
8.5 Backing Up Your Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
8.6 Common Errors and Fixes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16


**9** **Legal** **Notice** **17**


3


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Project** **Overview**


The Flight Pricing Engine is a command-line Python application that helps a licensed pilot calculate
and dynamically price seat costs for personal flights. Given inputs about a proposed flight — origin,
destination, passengers, weights, fuel prices, and duration - the system computes a data-driven
ticket price that covers operating costs and optimizes expected revenue based on historical demand
patterns.


The system improves over time. Every pricing session that ends in an accept or deny outcome is
stored, and two machine learning models update automatically:


  - A **logistic** **regression** model that learns which prices passengers tend to accept on each
route and in each context.


  - A **Q-learning** **agent** (Markov Decision Process) that develops a long-run pricing policy by
treating each session as a step in a sequential decision problem.


These models are blended with a **constrained** **nonlinear** **optimizer** that maximizes expected
revenue subject to hard aircraft safety constraints (weight and balance, fuel capacity) on every
single quote.





**Features**


  - **Weight** **&** **Balance** **check**  - automatically verifies that the proposed configuration (fuel +
passengers + payload + pilot) does not exceed the aircraft’s maximum usable weight before
any pricing is computed. No CG (Center of Gravity) calculations are performed as of this
version 2.2 of the README or version 1.3 of the software.


  - **Fuel** **LP** **optimization**  - solves a small linear program to determine whether it is cheaper
to fuel fully at origin or split fueling across origin and destination, accounting for per-gallon
prices at both airports.


  - **Dynamic** **ticket** **pricing**  - computes the revenue-maximizing price per passenger subject
to a cost-recovery floor (minimum 5% margin) and a configurable market ceiling.


  - **Online** **learning**  - logistic regression coefficients and Q-table values update after every
session. The system does not require a batch retraining step.


  - **Persistent** **model** **state**  - learned parameters survive program restarts via two JSON files
that are overwritten (not appended) after each session.


  - **Analytics** **report**  - a built-in report shows per-route acceptance rates, average prices, total
revenue, logistic regression coefficients, and Q-agent policy statistics.


  - **No** **external** **API** **required**  - all computation is local; no internet connection is needed
at runtime.


4


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**System** **Requirements**


**Python** **Version**


Python **3.8** **or** **higher** is required. The system uses only `f-strings`, `pathlib`, and `typing` features
available since Python 3.8.


**Required** **Libraries**


Only two third-party libraries are needed beyond the Python standard library:


**Library** **Version** **Purpose**


`numpy` _≥_ 1.20 Price grid construction, vectorized
probability evaluation
`scipy` _≥_ 1.6 Brent scalar optimizer for price
refinement


All other imports ( `json`, `math`, `random`, `datetime`, `pathlib`, `typing` ) are part of the Python standard library and require no installation.


**Getting** **Started:** **Setup** **Guide**


This section walks through every step required to go from a fresh machine to a running instance of
the pricing engine. Instructions are provided for Windows, macOS, and Linux. Set up instructions
for IDEs are also included.


**Step** **1** **—** **Verify** **Python** **is** **Installed**


Open a terminal (Command Prompt or PowerShell on Windows; Terminal on macOS or Linux)
and run:

```
 python --version

```

You should see output like `Python` `3.11.4` . If you see `Python` `2.x` or an error, install Python 3
from `[https://www.python.org/downloads/](https://www.python.org/downloads/)` .





**Step** **2** **—** **Download** **the** **Project** **Files**


Create a folder on your computer where you want the project to live. For example:


5


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


Place all nine `.py` files into this folder. When done, the folder should look exactly like this - no
subfolders needed yet (the `data/` directory is created automatically on first run):





**Step** **3** **—** **Create** **a** **Virtual** **Environment** **(Recommended)**


A virtual environment keeps the project’s dependencies isolated from your system Python installation. This is not strictly required, but it is strongly recommended.


You will know the environment is active when your terminal prompt shows `(venv)` at the beginning.
To deactivate it later, type `deactivate` .


**Step** **4** **—** **Install** **Dependencies**


With the virtual environment active, install `numpy` and `scipy` :

```
 pip install numpy scipy

```

To confirm the installation succeeded:

```
 python -c "import numpy, scipy; print(’Dependencies OK ’)"

```

You should see: `Dependencies` `OK` .


**Step** **5** **—** **Configure** **Your** **Aircraft** **Parameters**


The software is programmed with parameters for a specific PA-28-236 Piper Dakota aircraft. Before
running the system for the first time, open `config.py` in any text editor and update the values in
the _Aircraft_ _Limits_ and _Pilot_ sections to match your actual aircraft and weight.


6


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


The most important values to change are:


All other settings (pricing bounds, model hyperparameters) can be left at their defaults initially
and tuned later. See Section 7 for a full description of every configuration option.


**Step** **6** **—** **Run** **the** **Application**


From inside the `flight` ~~`p`~~ `ricer/` folder, run:

```
 python main.py

```

You will be greeted with a menu:





Select `1` to price your first flight. The `data/` folder and both JSON files will be created automatically
at this point.





**Setting** **Up** **in** **an** **IDE**


**Visual** **Studio** **Code**

1. Open VS Code and select _File_ _→_ _Open_ _Folder_, then choose your `flight` ~~`p`~~ `ricer/` folder.


2. Install the _Python_ extension by Microsoft (search “Python” in the Extensions panel).


3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), type _Python:_ _Select_ _Interpreter_, and
choose the interpreter from your virtual environment ( `venv` ).


4. Open `main.py` and press `F5` to run, or click the _Run_ button in the top-right corner.


5. The integrated terminal will appear at the bottom and the application menu will display
there.


7


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**PyCharm**

1. Select _File_ _→_ _Open_ and choose the `flight` ~~`p`~~ `ricer/` folder.


2. Go to _File_ _→_ _Settings_ _→_ _Project_ _→_ _Python_ _Interpreter_ . Click the gear icon and select _Add_
_Interpreter_ _→_ _Existing_ _Environment_, then point it to the Python executable inside your `venv`
folder.


3. Right-click `main.py` in the project tree and select _Run_ _’main’_ .


4. PyCharm will run the script in its integrated terminal at the bottom of the screen.





**Using** **the** **Application**


**Pricing** **a** **Flight** **—** **Input** **Reference**


When you select option `1` from the main menu, the system will prompt for the following inputs in
order:


Table 1: Input prompts and expected values.


**Prompt** **Format** **Notes**


Origin airport Text ICAO (e.g. `KCMI` ), IATA, or
plain name
Destination airport Text Same formats as origin
Departure date & time `YYYY-MM-DD` Used to compute booking
`HH:MM` lead time
Round trip? `y` / `n` Affects fuel LP and tach-hour
calculation
Number of passengers Integer Excludes pilot
Average passenger weight Decimal (lbs) Used for weight & balance
check
Total payload weight Decimal (lbs) Baggage and cargo;
maximum 200 lbs
Fuel price at origin Decimal ($/gal) Used in fuel LP objective
Fuel price at destination Decimal ($/gal) Used in fuel LP objective
Flight duration (one way) Decimal (hours) Tach-hour cost and fuel burn
are derived from this


**Reading** **the** **Output**


After computing, the system displays five result sections:


1. **Weight** **&** **Balance**  - total ramp weight with a component breakdown (fuel, passengers,
payload, pilot) and the margin below the max usable weight limit. If overweight, the session


8


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


stops here.


2. **Fuel Recommendation** **(LP Solution)**  - the optimal fueling strategy (origin-only, originfull, or split), gallons at each stop, and total fuel cost.


3. **Operating** **Costs**  - tach hours, aircraft cost, fuel cost, total operating cost, and cost per
seat.


4. **Model** **Status**  - how many SGD updates the logistic regression has processed, how many
Q-learning episodes the agent has experienced, the current exploration rate _ε_, and the recent
route acceptance rate.


5. **Ticket** **Pricing**  - the NLP-optimal price, the blended final price, estimated acceptance
probability _P_ (accept), expected revenue, and expected profit.


**Recording** **the** **Outcome**


After reviewing the price, you will be asked:

```
  Was this pricing accepted by the passenger(s)? (y/n):

```

Enter `y` if the passengers agreed to the quoted price, or `n` if they declined. **This** **step** **is** **critical**

- it is the feedback that trains both machine learning models. Never skip it, even for hypothetical
sessions.


**Viewing** **the** **Analytics** **Report**


Select option `2` from the main menu to see:


  - Overall totals: quotes, accepted, denied, total profit.


  - Per-route breakdown: acceptance rate, average price, average revenue, total revenue.


  - Current logistic regression coefficients with direction indicators.


  - Q-agent episode count, current _ε_, and a sample of non-zero Q-values from the policy table.


**Project** **Structure**


9


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Data** **Flow** **Summary**


1. `main.py` receives user menu selection and delegates.


2. `pricer.py` collects inputs and orchestrates the session.


3. `optimizer.py` solves the fuel LP and weight check, then runs the NLP price optimizer using
probabilities from `demand` `model.py` .


4. `mdp` `agent.py` provides a Q-learning price suggestion that is blended with the NLP result.


5. `history.py` retrieves the route acceptance rate (used in state construction) and persists the
completed record.


6. After outcome entry, `demand` `model.py` and `mdp` ~~`a`~~ `gent.py` both update their internal parameters and overwrite `model` ~~`s`~~ `tate.json` .


**Confguration** **Reference**


All tunable parameters live in `config.py` . The file is organized into six groups. See the following
tables for a description of each tunable parameter as well as the default values selected (you may
adjust all values to fit you best)


**Aircraft** **Limits**


**Constant** **Default** **Description**


`SEATS` ~~`A`~~ `VAILABLE` 3 Paying passenger seats
(total minus pilot)
`MAX` `USABLE` ~~`W`~~ `EIGHT` 1241.23 Combined weight limit
in lbs
`MAX` `FUEL` ~~`G`~~ `ALLONS` 72.0 Usable fuel capacity in
gallons
`FUEL` `LBS` ~~`P`~~ `ER` ~~`G`~~ `ALLON` 6.0 Avgas weight (lbs per
gallon)
`FUEL` `BURN` ~~`G`~~ `PH` 12.0 Cruise fuel burn
(gal/hr)
`MAX` `PAYLOAD` `LBS` 200.0 Hard baggage/cargo
limit (lbs)
`RESERVE` ~~`H`~~ `OURS` 0.75 VFR/IFR fuel reserve
requirement (hours)


10


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Pilot**


**Constant** **Default** **Description**


`PILOT` ~~`W`~~ `EIGHT` `LBS` 160 Pilot weight included
in W&B calculation


**Operating** **Costs**


**Constant** **Default** **Description**


`COST` `PER` ~~`T`~~ `ACH` ~~`H`~~ `OUR` 115.00 Aircraft cost per tach
hour ($)
`TACH` `TO` `FLIGHT` `HOUR` ~~`R`~~ `ATIO` 0.80 Long-run
tach-to-flight-hour
conversion


**Pricing** **Bounds**


**Constant** **Default** **Description**


`MIN` `MARGIN` 0.05 Minimum margin above
per-seat cost (5%)
`PRICE` ~~`C`~~ `EILING` ~~`M`~~ `ULTIPLIER` 5.0 Max price = 5 _×_
per-seat cost
`PRICE` ~~`G`~~ `RID` ~~`S`~~ `TEPS` 200 Price grid resolution for
NLP sweep


**Logistic** **Regression** **Hyperparameters**


**Constant** **Default** **Description**


`LR` `LEARNING` ~~`R`~~ `ATE` 0.05 SGD step size _α_
`LR` `REGULARIZATION` 0.01 L2 penalty
coefficient _λ_
`LR` `MIN` ~~`S`~~ `AMPLES` 5 Records before
model activates
`LR` `INITIAL` `BETAS` `[2.0,` `-0.01,` Initial coefficient
`-0.1,` vector
```
                      0.005,
                     -0.05, 0.1]

```

11


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**MDP** **/** **Q-Learning** **Hyperparameters**


**Constant** **Default** **Description**


`MDP` `DISCOUNT` `FACTOR` 0.92 Future reward discount
_γ_
`MDP` `LEARNING` `RATE` 0.15 Bellman update step
size _αQ_
`MDP` `EPSILON` `START` 0.25 Initial exploration rate
_ε_ 0
`MDP` `EPSILON` `MIN` 0.02 Exploration rate floor
`MDP` `EPSILON` `DECAY` 0.97 Per-episode decay
multiplier
`MDP` `PRICE` ~~`B`~~ `INS` 12 Discrete price
multiplier actions
`MDP` `STATE` ~~`B`~~ `INS` `[4,4,3,4]` State space bin counts


**Developer** **Guide**


This section is intended for contributors or anyone who wants to modify, extend, or debug the
codebase.





**Architecture** **Principles**


The codebase follows three design rules that should be preserved in any extension:


1. **Maintain** **strict** **layer** **separation.** `demand` `model.py`, `optimizer.py`, and `mdp` ~~`a`~~ `gent.py`
do not import from each other. They communicate only through `pricer.py`, which acts as
the orchestrator. If you add a new model layer, add it as a new module and wire it through
`pricer.py` .


2. **Config.py** **is** **the** **single** **source** **of** **truth.** No reused numbers appear in any module other
than `config.py` . If you find yourself hardcoding a threshold or weight value anywhere else,
move it to `config.py` first.


3. **Persistence is explicit.** Both `demand` ~~`m`~~ `odel.py` and `mdp` `agent.py` call their own ~~`s`~~ `ave` ~~`s`~~ `tate()`
methods immediately after updating parameters. Do not save state from `pricer.py` or any
other module.


12


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Module-by-Module** **Reference**

```
config.py

```

Contains only constants. No functions, no classes, no logic. Import individual constants using
explicit names:

```
 from config import MAX_USABLE_WEIGHT, FUEL_BURN_GPH

```

Never import `config` as a module object; this makes dependency tracking harder.

```
demand model.py

```

Defines the `LogisticDemandModel` class. Key methods:


**Method** **Description**


`predict` ~~`a`~~ `ccept` ~~`p`~~ `rob(.)` Returns _P_ (accept) _∈_ [0 _,_ 1] for a single
price/feature set. Returns 0.5 if fewer
than `LR` ~~`M`~~ `IN` ~~`S`~~ `AMPLES` updates have
occurred.
`predict` ~~`a`~~ `ccept` ~~`p`~~ `rob` `curve(grid,` `.)` Vectorized version over a numpy price
array. Used by the optimizer.
`update(.,` `accepted:` `bool)` Performs one SGD step and saves state.
Call this after every session.
`summary()` Returns a dict with current betas,
update count, and human-readable
interpretation. Used by `reports.py` .

```
optimizer.py

```

Contains only pure functions (no classes, no state). Key functions:


13


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Function** **Description**


`solve` ~~`f`~~ `uel` ~~`l`~~ `p(.)` Evaluates the two closed-form LP
strategies and returns the cheaper one as
a dict with keys `strategy`,
`gallons` `origin`, `gallons` `dest`,
`total` ~~`f`~~ `uel` `cost`, `note` .
`check` ~~`w`~~ `eight` `constraint(.)` Returns a dict with `feasible` (bool),
`total` ~~`l`~~ `bs`, `slack` ~~`l`~~ `bs`, and `breakdown` .
`max` ~~`g`~~ `allons` `within` ~~`w`~~ `eight(.)` Computes _G_ wb   - the maximum gallons
loadable without violating the weight
limit. Pass this to `solve` `fuel` ~~`l`~~ `p` .
`compute` ~~`o`~~ `perating` `cost(.)` Returns tach hours, aircraft cost, fuel
cost, and total operating cost as a dict.
`optimize` `price(.)` Main pricing function. Accepts a
`LogisticDemandModel` instance and
returns the full pricing result dict
including `final` `price` ~~`p`~~ `er` ~~`p`~~ `ax`,
`expected` ~~`r`~~ `evenue`, and the raw
price/revenue curve arrays.

```
mdp a gent.py

```

Defines the `QLearningAgent` class and several standalone discretization functions. Key components:


**Component** **Description**


`build` ~~`s`~~ `tate(.)` Takes acceptance rate, lead days, num
passengers, and departure month; returns
a `(int,` `int,` `int,` `int)` state tuple.
`action` ~~`t`~~ `o` ~~`m`~~ `ultiplier(k)` Maps action index _k_ _∈_ [0 _, NA −_ 1] to a
price multiplier in [0 _._ 60 _,_ 1 _._ 40].
`suggest` ~~`p`~~ `rice` ~~`m`~~ `ultiplier(state)` Returns the greedy multiplier and mode
string ( `"exploit"` ) with no exploration.
Use this for the final price suggestion in
`pricer.py` .
`select` ~~`a`~~ `ction(state)` Returns an _ε_ -greedy action for training.
`record` ~~`o`~~ `utcome(s,` `a,` `r,` `s’)` Applies the Bellman update and saves
state.
`policy` ~~`s`~~ `ummary()` Returns the full greedy policy table as a
dict. Used by `reports.py` .


14


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_

```
history.py

```

Defines the `FlightHistory` class. Key methods:


**Method** **Description**


`append(record:` `dict)` Appends a session record to
`flight` ~~`h`~~ `istory.json` .
`all` ~~`r`~~ `ecords()` Returns the full list of session records.
`records` ~~`f`~~ `or` `route(key)` Filters records to a specific route key
(e.g. `KCMI-KORD` ).
`route` ~~`a`~~ `cceptance` `rate(key)` Returns the acceptance rate for the most
recent 20 records on a route. Returns 0.5
if fewer than 2 records exist.
`compute` ~~`l`~~ `ead` `days(date` ~~`s`~~ `tr)` Parses a date string and returns days
until departure.
`route` ~~`s`~~ `ummary()` Returns per-route aggregate statistics for
`reports.py` .

```
pricer.py

```

The `run` ~~`p`~~ `ricing` ~~`s`~~ `ession()` function is the only public interface. It instantiates all three model
classes, runs the 14-step session flow, and returns nothing (all side effects are persistence). If you
want to add a new input field, this is the only file you need to touch in addition to `config.py` and
the relevant model file.

```
utils.py

```

Provides `prompt` ~~`f`~~ `loat()`, `prompt` ~~`i`~~ `nt()`, `prompt` ~~`s`~~ `tr()`, `prompt` ~~`b`~~ `ool()`, `format` `currency()`,
`print` `header()`, `print` ~~`s`~~ `ection()`, `print` ~~`w`~~ `arning()`, `print` ~~`o`~~ `k()`, and `print` ~~`i`~~ `nfo()` . These
are pure I/O helpers with no business logic. Edit this file only to change the CLI appearance.


**Adding** **a** **New** **Feature** **to** **the** **Model**


To add a new input feature to the logistic regression (for example, day of week), follow these steps:


1. In `config.py`, add a new initial beta value to `LR` ~~`I`~~ `NITIAL` ~~`B`~~ `ETAS` (e.g. `0.02` ).


2. In `demand` ~~`m`~~ `odel.py`, update ~~`b`~~ `uild` `feature` `vector()` to include the new feature as an
additional element in the returned list. Update the `feature` `names` list in `summary()` to
match.


3. In `pricer.py`, add a prompt to collect the new input and pass it through to `lr` ~~`m`~~ `odel.update()`
and `demand` `model.predict` ~~`a`~~ `ccept` ~~`p`~~ `rob()` .


4. **Delete** `data/model` ~~`s`~~ `tate.json` before the next run to reset the model with the new coefficient vector. The old betas have the wrong dimensionality and will cause errors.





15


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Resetting** **the** **Models**


To reset one or both learned models without losing flight history:


  - **Reset** **both** **models:** Delete `data/model` ~~`s`~~ `tate.json` . The file will be recreated from
`LR` `INITIAL` `BETAS` and a zeroed Q-table on the next run.


  - **Reset only the logistic regression:** Open `model` `state.json` and delete the `"logistic` ~~`r`~~ `egression"`
key, then save.


  - **Reset** **only** **the** **Q-agent:** Open `model` ~~`s`~~ `tate.json` and delete the `"q` ~~`l`~~ `earning"` key, then
save.


  - **Reset** **everything** **including** **history:** Delete the entire `data/` folder. Both files will be
recreated on next run.


**Backing** **Up** **Your** **Data**


The `data/` folder contains everything the system has learned. Back it up regularly, especially
`flight` `history.json` . A simple approach is to copy the folder with a date suffix:


**Common** **Errors** **and** **Fixes**


**Error** **Fix**


`ModuleNotFoundError:` `numpy` Run `pip` `install` `numpy` `scipy` with
your virtual environment active.
`ModuleNotFoundError:` `config` Run the script from inside the
`flight` ~~`p`~~ `ricer/` folder, not from a
parent directory.
`JSONDecodeError` on startup One of the JSON files in `data/` is
corrupt. Delete `model` ~~`s`~~ `tate.json` (and
`flight` ~~`h`~~ `istory.json` if needed) and
restart.
`ValueError` from numpy in optimizer Usually caused by a mismatched beta
vector length. Delete `model` `state.json`
and restart.
Input not being read (IDE) Enable interactive terminal mode in your
IDE. See Section 4.2.


16


_Flight_ _Pricing_ _Engine_ _README_ _&_ _Developer_ _Guide_


**Legal** **Notice**



Flight Pricing Engine - README & Developer Guide


17


