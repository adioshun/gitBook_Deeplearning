# [Catastrophic Forgetting](http://standoutpublishing.com/g/catastrophic-forgetting.html)


---

# [What is catastrophic forgetting and how does it affect AI development?](https://www.techworld.com/tech-innovation/what-is-catastrophic-forgetting-how-does-it-affect-ai-development-3687007/)

> 2018.11


People who suffer from memory loss could find kindred spirits in machines afflicted with a condition known as "catastrophic forgetting".

The affliction is caused when neural networks are trained to complete new tasks.


These systems learn by analysing training data that helps them interpret future information, whether they're facial recognition software scanning profile photos to learn how to detect humans in a video, or self-driving cars studying obstacles on the road to understand what they need to avoid.

When they're taught a new task they have to adapt their parameters to solve it. These can overwrite the parameters that they learned before and lead them to "forget" the knowledge that they previously acquired.

The phenomenon is one of the biggest barriers to creating an artificial general intelligence (AGI) as it eliminates the capacity for continual learning, a process of learning new tasks in a sequence.

"This artificial general intelligence should have this capability of sequential learning in a changing world as it's always going to find new things that change and it should adapt to these changes," says Joan Serrà, a research scientist specialising in deep learning and machine learning at Telefónica R&D. "And if by adapting to these changes it forgets then it's not intelligent anymore."

Computer scientists have been tackling the challenge for decades. Serrà's research team may have found a solution.

## Memory improvement
The Telefónica research team in Barcelona have developed a system that addresses the problem of catastrophic forgetting by minimising the amount of memory that the algorithm requires.

Their mechanism consists of two separate parameters. One compacts the information that the neural networks require into the fewest neurons possible without compromising its accuracy, while the other protects the units that were essential to complete past tasks.

"Our solution tries to minimise the space while learning a new task and at the same time not changing what was learnt before in this reduced space where you put this knowledge," says Serrà.

The research turned established ideas into a functioning system.

"We all know that if we don't touch the neurons they don't forget," says Serrà. "And also there is a trend in neural networks and machine learning of trying to learn very compact models for deploying these super-advanced neural networks in resource-poor machines like a car or a mobile phone or machines that have very little memory or capacity, so you want to compress these neural networks. There were these two things and we just combined them in a clever way."

His team now hopes to use this compression capability to deploy advanced algorithms in resource-poor devices such as cars or mobile phones.

"We think that part of the techniques that were used can be recycled for deploying these algorithms," he says.

## Alternative systems
Telefonica's R&D team is not the only organisation that has developed a solution to catastrophic forgetting. Google DeepMind researchers recently created an AI agent that can imagine what something found in one virtual environment may look like in another.

The system draws on five examples of how an object looks from different angles to learn what it is and how it might look from other angles or in different lighting. It can imagine how it would look in a different virtual world and differentiate between that and what it had previously seen, allowing it to remember the learning that preceded it.

Telefónica's system may not be the first Serrà believes it is the best.

Many of the alternatives only tested their systems with limited quantities and types of data and then artificially created tasks from these datasets, while Telefonica's team tested different datasets for different tasks, from identifying handwritten digits to recognising faces.


"It's when you put the algorithms in this more rigorous evaluation setup that you see that some of them are really underperforming and then there is still room for improvement. They are still forgetting something," says Serrà.

## Remaining barriers
Serrà admits that the Telefónica system alone will not cure catastrophic forgetting.

He believes that they could still make more efficient use of the units of information used by neural networks, but this alone would not solve the fundamental issue of training neural networks using backpropagation. The problem with this method is that it optimises a network for a fixed target, and that could be disrupted by inputting new information.

Serrà is convinced that further compression is possible but believes that the ultimate solution would be an entirely new method of remembering that no longer relies on backpropagation.

"I would say that the key thing would be to find a training algorithm that's maybe more biologically inspired and that is not based on erasing past information," he says. "But that is a huge thing."