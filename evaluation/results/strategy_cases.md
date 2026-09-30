# So sanh 4 cau hinh tren cung truy van

Rerank: CLIP ViT-L/14, N=50. Moi o la caption dau cua anh top-1..3.

## "two Dobermans playing on a lawn"

| cau hinh | #1 | #2 | #3 | ms |
|---|---|---|---|---|
| BM25 | Two Dobermans fighting or playing together on a lawn . | A boy and girl playing with buckets in a lawn sprinkler | A white dog runs past another dog who is standing besid | 3 |
| dense | Two Dobermans fighting or playing together on a lawn . | The dogs are in the snow in front of a fence . | A dog with a Frisbee in front of a brown dog . | 14 |
| hybrid | Two Dobermans fighting or playing together on a lawn . | The dogs are in the snow in front of a fence . | A black dog is running along the beach beside a brown d | 16 |
| hybrid+rerank | Two Dobermans fighting or playing together on a lawn . | The dogs are in the snow in front of a fence . | The two white dogs are playing in a field . | 25 |

## "a dog catching a frisbee"

| cau hinh | #1 | #2 | #3 | ms |
|---|---|---|---|---|
| BM25 | A brown and white dog with a Frisbee in his mouth . | A black and white dog is catching a Frisbee in the yard | A dog with a Frisbee in front of a brown dog . | 2 |
| dense | A black and white dog is going after an orange Frisbee  | A dog is jumping in the air trying to catch a red Frisb | A dog with a Frisbee in front of a brown dog . | 10 |
| hybrid | A black and white dog is going after an orange Frisbee  | A dog with a Frisbee in front of a brown dog . | A dog is jumping in the air trying to catch a red Frisb | 11 |
| hybrid+rerank | A black and white dog is going after an orange Frisbee  | A dog with a Frisbee in front of a brown dog . | A dog is jumping in the air trying to catch a red Frisb | 25 |

## "someone riding a board with wheels down the street"

| cau hinh | #1 | #2 | #3 | ms |
|---|---|---|---|---|
| BM25 | A man surfing a wave . | A snowboarder is doing a flip over a mound of snow . | A man in an orange shirt flips his skateboard on a subu | 3 |
| dense | A man in an orange shirt flips his skateboard on a subu | A man wearing a red helmet jumps up while riding a skat | A barefoot and shirtless skateboarder rides along a roa | 10 |
| hybrid | A man in an orange shirt flips his skateboard on a subu | A boy in a blue shirt jumps down stairs on a skateboard | A man does a trick on his skateboard in front of a crow | 12 |
| hybrid+rerank | A man in an orange shirt flips his skateboard on a subu | A barefoot and shirtless skateboarder rides along a roa | A man on a motorcycle riding on his back tire . | 25 |

## "kids splashing around in a pool"

| cau hinh | #1 | #2 | #3 | ms |
|---|---|---|---|---|
| BM25 | A few younger boys play around a fountain . | A black and white dog is playing in a pond or creek . | A dog comes out of a blue baby pool with a ball in his  | 2 |
| dense | Two girls running through pool while water sprays over  | A boy wearing red shorts is jumping into a paddling poo | A toddler is making a splash inside a blue paddling poo | 9 |
| hybrid | A toddler is making a splash inside a blue paddling poo | A bunch of dogs and kids playing in a swimming pool . | A child slides down a slide and into the water . | 11 |
| hybrid+rerank | A child slides down a slide and into the water . | A boy with a beach ball behind him playing in a pool | A boy is making a splash in a swimming pool . | 21 |

## "a street with no cars at all"

| cau hinh | #1 | #2 | #3 | ms |
|---|---|---|---|---|
| BM25 | A man jumps train cars with his bike . | A race cars muffler catches on fire . | A man sitting on the left near a man walking along the  | 3 |
| dense | A busy city street with many people walking along it . | a couple walking down the street carrying bags | A man in the near distance is pointing to a silver seda | 9 |
| hybrid | A man sitting on the left near a man walking along the  | A busy city street with many people walking along it . | a couple walking down the street carrying bags | 11 |
| hybrid+rerank | A man wearing a black hat walks along a road next to a  | A man in the near distance is pointing to a silver seda | A couple are sitting on a rock in front of a street . | 22 |

## "a man in a red shirt climbing a rock"

| cau hinh | #1 | #2 | #3 | ms |
|---|---|---|---|---|
| BM25 | A man rock climbs up a very tall cliff , with a blue ro | a man climbs a rock . | A man repels down a cliff over water . | 3 |
| dense | A boy climbing a rocky area . | A rock climber repels off a rocky wall . | A dark-haired man wearing a brown shirt is free-climbin | 8 |
| hybrid | A man rock climbs up a very tall cliff , with a blue ro | A dark-haired man wearing a brown shirt is free-climbin | A man in a pink shirt climbs a rock face | 10 |
| hybrid+rerank | A man climbing a mountain . | A man dressed in grey climbing a large brown rock . | One climber is sitting on the rock overlooking the ocea | 22 |
