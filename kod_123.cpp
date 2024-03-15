for (auto& threadData : threadDataVec) {
  lat_sub.insert(lat_sub.end(), std::make_move_iterator(threadData.lat_sub.begin()), std::make_move_iterator(threadData.lat_sub.end()));
  lon_sub.insert(lon_sub.end(), std::make_move_iterator(threadData.lon_sub.begin()), std::make_move_iterator(threadData.lon_sub.end()));
  insu_sub.insert(insu_sub.end(), std::make_move_iterator(threadData.insu_sub.begin()), std::make_move_iterator(threadData.insu_sub.end()));
  reas_sub.insert(reas_sub.end(), std::make_move_iterator(threadData.reas_sub.begin()), std::make_move_iterator(threadData.reas_sub.end()));
  premium_sub.insert(premium_sub.end(), std::make_move_iterator(threadData.premium_sub.begin()), std::make_move_iterator(threadData.premium_sub.end()));
}
