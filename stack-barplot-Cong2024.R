# library
library(ggplot2)
library(tidyr)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

getwd()
df = read.csv('/Users/data.csv')
df_long = gather(df, Cmp_type, Percentage, 
                 Phrasal:ER, 
                 factor_key = TRUE)
p = ggplot(df_long, aes(fill=Cmp_type, y=Percentage, x=level_id)) + 
  geom_bar(position="fill", stat="identity")  +
  xlab(label = 'Proficiency level id') +
  ylab(label = 'Percentage out of total Cmp') + 
  theme(panel.background = element_rect(fill = "white", colour = "grey50"),
        legend.text = element_text(size=14, face="bold"),
        legend.title=element_text(size=14, face="bold"),
       
        axis.title.x = element_text(size=20,face="bold"),
        axis.text.x = element_text(face="bold", color="#993333", 
                                   size=12, angle=360),
        
        axis.title.y = element_text(size=20,face="bold"),
        axis.text.y = element_text(face="bold", color="#993333", 
                                   size=12, angle=360)) +

  scale_fill_manual("Cmp_type", values = c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442"))

p
ggsave('stack_barplot.png', p, 
       dpi = 300)
