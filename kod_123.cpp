for (const auto& threadData : threadDataVec) {
  lat_sub.insert(lat_sub.end(), threadData.lat_sub.begin(), threadData.lat_sub.end());
  lon_sub.insert(lon_sub.end(), threadData.lon_sub.begin(), threadData.lon_sub.end());
  insu_sub.insert(insu_sub.end(), threadData.insu_sub.begin(), threadData.insu_sub.end());
  reas_sub.insert(reas_sub.end(), threadData.reas_sub.begin(), threadData.reas_sub.end());
  premium_sub.insert(premium_sub.end(), threadData.premium_sub.begin(), threadData.premium_sub.end());