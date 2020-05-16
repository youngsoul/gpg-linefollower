#//# --------------------------------------------------------------------------------------
#//# Created using Sequence Diagram for Mac
#//# https://www.macsequencediagram.com
#//# https://itunes.apple.com/gb/app/sequence-diagram/id1195426709?mt=12
#//# --------------------------------------------------------------------------------------
participant GoPiGo as g
participant MacBook_ImageHub as i

loop [while collecting images]
g->i: send image
== Person reviews image to decide label ==
i-->i(---:>): determine the label
i-->i(---:>): save image to training folder
opt to delete the image 
i-->i(---:>): d - delete last saved image
i->g: Send Ok, no command
end
== End Person Review ==
i->g: send turn label
group Turn label one of the following
      i-->g: 1 - turn left
      i-->g: 2 - straight
      i-->g: 3 - turn right
      i-->g: 0 - stop
      i-->g: 9 - forward
      i-->g: 8 - backwards
      i-->g: x - exit GoPiGo client
      i-->g: d - turn right
      
      

end
g->g: perform turn instruction
g->g: move forward small amoumnt
end


