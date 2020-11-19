# Meeting Notes

## 2020/11/17

* That we want NBT engaged is clear to me, which is what motivated me
  to try to present/visualize the work in a way that could bring
  others along with us.
* However, I had not really thought about what we help we should ask
  them for. In what form could we accept feedback?
* We need to find a way to engage them for ideas of what we could
  cluster that might be helpful. This gets to Anthony's workshop idea,
  but we should do some thinking well before that.
* I am envisioning a spreadsheet/form where we could ask for 
  1. a variable to cluster, this could be some direct output like gpp
     or something that we may have to think about how to estimate from
     model output.
  2. the time frame to stack. For example, we did seasonal/decadal
     averages for the gpp comparison we showed. This will vary from
     variable to variable.
  3. the other variables to include in the cluster, most probably some
     aspects of variables which control the primary variable to
     cluster. In the case of our run, this was tas and pr.
* Then we could run each of these solo, like the gpp run, to get an
  understanding of individual effects with a view to how to combine
  them in a more complete clustering.
* One doubt I heard expressed related to processes not represented in
  the model. It may be that ecosystems are most vulnerable in a
  dimension not represented in any model--our analysis won't possibly
  capture that.
* Another doubt was related to being able to express confidence in our
  assessment of what is vulnerable. I feel that the use of multiple
  models can partly do this, but the comment may have come more from a
  place of skepticism that models are useful. Ultimately we will make
  recommendations and they will have to assess if they should be
  believed.

### Zoom Chat Dump

## 2020/11/6

### Visualization Feedback

* Contribution plots should be in terms of areas
* The entire visualization needs more labeling
* Coloring should be made consisitent
* Needs more linkage, fade the map when things are selected
* Could we use the size of the markers in the parametric plots to
  indicate something, perhaps area contribution at the given time
  indicated? Or it could be the within-cluster variance
* Need a view which shows things from a single cluster point of view
  as well. What are the distributions of values in a cluster? At what
  times do we see the clusters?
* We want to keep the time range back to 1850 even if we later only
  show vulnerability across a range in the modern era. This is to
  better understand what has changed and define regions in their
  historical context

## 2020/10/12

* Jitu/Nate will pursue climate driver clustering, Jitu already has
  some work done on this
* Moet/Deeksha/Min will begin an evaluation of temperature and
  precipitation extremes of CMIP6 models
  - Some of this evaluation they have done in another context, hoping
    that the assessment gives us some direction
  - Nate will run ILAMB through the CMIP6 models available to give us
    an idea of what is out there
* We will target OLCF for now. Current allocations there will allow us
  to work via Rhea and we can write a DD for a formal allocation, Nate
  will look into this.
* Visualizations shown via Dash/Plotly are good, could also look at
  Holoviews/Geoviews
  - For requirements for serving up the apps, could ask Sarat
* The Next Big Thing (NBT) is an effort that many in ESD area already
  involved in, responsible for working with Stan to get the Vulnerable
  Ecosystems language in the LDRD call in the first place. They are a
  good group to engage to help get feedback.
  - Giving them a project overview on Nov 13, any results we can piece
      together is an opportunity for feedback
