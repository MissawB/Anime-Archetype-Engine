from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('switch_mode/<str:mode>/', views.switch_mode, name='switch_mode'),
    path('start_game/', views.start_game, name='start_game'),
    path('game/', views.game_view, name='game'),
    path('game/abandon/', views.abandon_game, name='abandon_game'),
    path('game/hint/<str:hint_type>/', views.reveal_hint, name='reveal_hint'),
    path('make_guess/', views.make_guess, name='make_guess'),
    path('archetypist/', views.archetypist_view, name='archetypist'),
    path('paradox/', views.paradox_view, name='paradox'),
    path('paradox/guess/', views.paradox_guess, name='intruder_guess'),
    path('undercover/party/setup/', views.undercover_party_setup, name='undercover_party_setup'),
    path('undercover/party/play/', views.undercover_party_play, name='undercover_party_play'),
]
