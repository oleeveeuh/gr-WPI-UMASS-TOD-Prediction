# Load relevant packages
library(tidyverse)

# read in data
patient_info_raw <- read.csv("data/GSE71620_Phenotype_GEO.csv")
samples_raw <- read.csv("data/all_sample.csv")
# note this file has been edited to get rid of the info at the top before the actual data
gene_info <- read.csv("data/gene_names.csv")

            # cause_death <- read.csv("data/cause_of_death.csv")
# from the file "pnas.1508249112.sd01.xlsx". Non-circadian rows are hidden which makes 
#converting the xlsx file to a csv difficult, so this is just copy-pasted from the xlsx sheet for ease
circadian_genes <- c("C1orf51" , "NR1D1" , "PER3" , "PER2" , "KCNH4" , "NR1D2" , "PER1" , "ARNTL" , "OPRL1" , "BHLHE41" , "DBP" , "DUSP11" , "PIAS1" , "PDZRN3" , "ADRA1B" , "RNF115" , "ENG" , "SPRY4" , "NPAS2" , "BHLHE40" , "NFIL3" , "LRRC39" , "LDB1" , "DUSP4" , "USP2" , "XRN1" , "KIAA1370" , "RARA" , "RSPO2" , "IRS2" , "RC3H1" , "ZNF43" , "ANKRD12" , "SESN3" , "AKAP1" , "TRIM24" , "SLCO2A1" , "SBK1" , "TRAF5" , "ALOX5AP" , "PDE7B" , "TMEM119" , "BACE2" , "RPS6KA5" , "ZBBX" , "FGD5" , "CCDC91" , "FHL3" , "PEX1" , "DCUN1D4" , "GRIK2" , "LCP1" , "SLC39A14" , "IGFBP4" , "THRA" , "CREBZF" , "IQCB1" , "ATG14" , "IFNGR1" , "TMED10" , "NET1" , "ANKRD55" , "REV1" , "RAI1" , "TRIB2" , "SDK1" , "TOX" , "MCF2L" , "MED12L" , "MIA3" , "MAPK4" , "PUS7" , "SIK2" , "KIAA1524" , "LPCAT3" , "ORC3" , "SYNPO" , "MYSM1" , "HPGDS" , "CMTM8" , "KIAA1549" , "CDH5" , "SH3BP1" , "PHF17" , "VGF" , "SNORD80" , "ABCA6" , "KIAA0776" , "INSIG1" , "ABHD2" , "NAB2" , "C19orf77" , "PDP2" , "EPN2" , "FEZF2" , "FBXO40" , "C5orf51" , "GPR83" , "PCDH8" , "CCDC14" , "GIMAP4" , "MPP7" , "ARHGAP25" , "C17orf76" , "FABP5" , "KLHDC1" , "TMEM41A" , "ZNF441" , "SCAMP2" , "TUBGCP5" , "CSF1R" , "CTNND1" , "STK35" , "KIAA1107" , "C15orf44" , "P2RY13" , "SEC11A" , "CHRM4" , "NXPH3" , "AFAP1L2" , "NRP1" , "MDGA1" , "SMARCA1" , "EXOC6B" , "ZFAND3" , "FAM81A" , "ZNF451" , "CD34" , "C20orf12" , "SLCO2B1" , "REPS2" , "EML4" , "SHC3" , "LYN" , "TARBP1" , "CTNNAL1" , "SIPA1L3" , "PYGO2" , "CEP57" , "LAPTM5" , "SNORD47" , "FAM45A" , "PCDH12" , "SAMD15" , "MGC42105" , "MPDZ" , "SLC39A11" , "FAM86B1" , "RASD1" , "MTIF3" , "MAP3K9" , "RPS6KC1" , "DICER1" , "FAM76A" , "MARCH7" , "ANKMY2" , "SRRM4" , "PPFIA4" , "DYNC2H1" , "GAL3ST4" , "VPS13B" , "SLC7A5" , "SIGLECP3" , "HABP2" , "JMY" , "CHD6" , "GLTSCR1" , "RBM26" , "ZNF488" , "SMPD3" , "ATPBD4" , "EEA1" , "HR" , "PLXNA1" , "OMG" , "FBXL18" , "PDE7A" , "KCNG1" , "TTLL5" , "TACC1" , "TCEAL2" , "KIAA1310" , "ZNF107" , "MYNN" , "ATR" , "POU3F1" , "LYPLA2" , "SDC3" , "ZNF678" , "UNC80" , "RARS2" , "EXOC5" , "KCTD12" , "NCRNA00208" , "PNRC2" , "PECAM1" , "AIF1" , "SLC43A3" , "STRADB" , "RSPH1" , "KIAA0125" , "FAM135A" , "RCL1" , "COG1" , "PPP2CB" , "ISLR2" , "HERC5" , "SPATA2L" , "SNRNP40" , "SH3GL3" , "PGS1" , "CAPN7" , "ASCL3" , "SRRT" , "RBM3" , "PTPRC" , "HECW2" , "TMED5" , "PRKCH" , "PRPF40A" , "MYH16" , "POLI" , "APOC2" , "UBR4" , "SNORD114-3" , "C9orf68" , "STAMBPL1" , "MARCH3" , "SERAC1" , "TCERG1L" , "WIPI1" , "PHTF2" , "DENND1B" , "SHISA6" , "CPLX1")

# get relevant info from patient data
patient_info <- patient_info_raw[!is.na(patient_info_raw$TOD), ]
patient_info['TOD_pos'] <- patient_info['TOD'] + 6

## Got rid of thise because we are not including cause of death in final analysis, but keeping code for later just in case
      # |> 
      #   # combine it with cause of death info, specify all other cols to join on to avoid accidental duplicates
      #   left_join(cause_death[, 2:9], by = c("Age" = "Age", "PMI" = "PMI", "pH" = "pH", "RIN" = "RIN", "Sex" = "Sex", "Race" = "Race" ))

# reassign datatypes
patient_info$ID <- as.factor(patient_info$ID)
gene_info$ID <- as.factor(gene_info$ID)


# pivot the sample data
samples <- data.frame(t(samples_raw[-1]))
colnames(samples) <- samples_raw[, 1] 
samples["sample_id"] <- rownames(samples)
#deal with column order
samples <- samples[, c(33298, 1:33297)]
#get rid of annoying rownames
rownames(samples) <- NULL

samples <- samples|> 
  pivot_longer(cols = matches("\\d+"), 
               names_to = "gene_ID", 
               values_to = "expression_level" ) |> 
  mutate(BA_ID = str_extract(sample_id, "(?<=_BA)\\d{2}(?=\\.)")) |> 
  mutate(patient_ID = str_extract(sample_id, "(?<=\\.)\\d{1,3}(?=\\.CEL)"))

# reassign data types
samples$gene_ID <- as.factor(samples$gene_ID)
samples$BA_ID <- as.factor(samples$BA_ID)
samples$patient_ID <- as.factor(samples$patient_ID)



full_df <- samples |> 
  left_join(patient_info[, c('ID', 'Age', 'Sex', 'TOD_pos')], by = c("patient_ID" = "ID")) |> 
  drop_na() |> 
  left_join(gene_info[, c("ID", "gene_assignment")], by = c("gene_ID" = "ID")) |> 
  mutate(gene_name = str_extract(gene_assignment, "(?<= // )[^ ]+(?= //)")) |>
  # remove rows where gene_assignment contains "---" (no assignment found)
  filter(gene_assignment != "---")

full_df <- full_df |> 
  select(-c(sample_id, gene_ID, gene_assignment)) |> ## Work Here!
  group_by(patient_ID, BA_ID, gene_name) |> 
  mutate(expression_level = max(expression_level)) |> 
  distinct() |> 
  filter(gene_name %in% circadian_genes) |> 
  ungroup() |> 
  select(-patient_ID)

full_df <- full_df |> 
  pivot_wider(names_from = gene_name, values_from = expression_level)

final_data_BA11 <- full_df[full_df$BA_ID == 11, -1] 
final_data_BA47 <- full_df[full_df$BA_ID == 47, -1]

write.csv(full_df[, -1], "data/wrangled data/full_data_6_17_2024.csv")
write.csv(final_data_BA11, "data/wrangled data/BA11_data_6_17_2024.csv")
write.csv(final_data_BA47, "data/wrangled data/BA47_data_6_17_2024.csv")





# DATA VISUALIZATION

tidy_full <- full_df |> 
  pivot_longer(cols = -c(BA_ID:Manner), names_to = 'gene_name', values_to = 'expression_level')


## Just look as a few different genes
ggplot(tidy_small[tidy_small$BA_ID == 11,], aes(x = TOD, y = expression_level)) +
  geom_point(color = "#ab1a2d")+
  facet_wrap(~ gene_name, scales = "free", dir = "v") +
  theme_light() +
  labs(title = "Gene Rhythmicity, BA 11", subtitle = "Using data from Chen et al.")

