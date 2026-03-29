# Documentation: Synthetic Check In Data Generation

The persona generator simulates realistic, complex human behaviour. This allows us to stress-test our model's sensitivity to subtle shifts in student well-being.



## 1. Student Archetypes 
**The Concept:** An Archetype is a "personality blueprint." It defines the baseline probability of how a student feels or behaves on an average day.
* **How it works:** We define a set of probabilities for features like `Emotion`, `Tiredness`, and `Eating`. For a `BASE_OPTIMIST`, we might set `HAPPY` to an 80% probability.
* **Why we use it:** It allows us to represent every type of student, including those who may not appear in initial training data. We can create "Stable Angry" students or "High-Intensity Anxious" students just by swapping the blueprint.



## 2. Temporal Modifiers 
**The Concept:** Real humans are not static; emotions fluctuate based on life events. The Temporal Modifier simulates a "Stress Trajectory" over time.
* **How it works:** We use mathematical functions to dictate how student stress changes across the semester.
    * **Linear:** A gradual, steady increase in pressure.
    * **Sigmoid:** A "tipping point" where a student is fine until a sudden, rapid descent into burnout.
    * **Sine Wave:** Cyclical stress (e.g., feeling worse on weekends or during midterms).
    * **Step Function:** A sudden, permanent shift (e.g., after a specific negative life event).
* **Why we use it:** This tests our model’s **velocity detection**. Can the model catch a "Sigmoid Burnout" faster than a "Linear Decline"?



## 3. The Variance Slider 
**The Concept:** Even the most stable student has "off" days. The Variance Slider prevents our data from being too robotic or predictable.
* **How it works:** It acts as an entropy dial. At **0% variance**, a student follows their archetype perfectly. At **100% variance**, their behaviour becomes completely random.
* **Why we use it:** It ensures our unsupervised model is robust enough to find patterns even when a student's data is messy or inconsistent.





## 4. Cross-Feature Correlation 
**The Concept:** Symptoms of distress rarely happen in isolation. If a student is highly stressed, they are likely tired and not eating well.
* **How it works:** We use a hidden **Latent Stress** variable. As this variable rises (driven by the Temporal Modifier), it automatically pulls the probabilities of negative features up. 
    * **High Stress** $\rightarrow$ Higher chance of `SAD` + Higher chance of `TIRED` + Higher chance of `NOT EATING`.
* **Why we use it:** This tests for **"Masked Risk."** If a student reports being `HAPPY` but their physical indicators (Eating/Tiredness) are crashing, can our risk engine still identify them as "At Risk"?